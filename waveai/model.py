import math

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from x_transformers import Decoder, MultiInputTransformerWrapper


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool,
        cross_attention: bool,
        rotary_emb: bool,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention
        self.decoder = True

        self.in_proj_weight = nn.Parameter(torch.empty((embed_dim * 3, embed_dim)))
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.key_cache = None
        self.value_cache = None

        if rotary_emb:
            rotary_dim = (
                embed_dim // num_heads
            ) // 2  # rotary embedding dim = dim of each head / 2
            self.rotary_emb = RotaryEmbedding(dim=rotary_dim)
        else:
            self.rotary_emb = None

    def forward(self, query, key, value, key_padding_mask=None):
        if not self.cross_attention:
            key = query
            value = query

        q = F.linear(query, self.in_proj_weight[: self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim : 2 * self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2 * self.embed_dim :])

        q, k, v = [
            x.reshape(
                x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads
            )
            for x in [q, k, v]
        ]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        B, _, T, _ = q.shape

        if self.rotary_emb is not None:
            # apply rotary embedding to queries and keys (if not cross-attention)
            q = self.rotary_emb.rotate_queries_or_keys(q)

            if not self.cross_attention:
                k = self.rotary_emb.rotate_queries_or_keys(k)

        x = flash_attention(q, k, v, causal=self.causal, padding_mask=key_padding_mask)
        x = x.view(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x, None


def get_causal_mask(size: int):
    queries_pos = torch.arange(size).view(-1, 1)
    keys_pos = torch.arange(size).view(1, -1)
    return torch.where(
        (queries_pos - keys_pos) >= 0, torch.zeros([]), torch.full([], float("-inf"))
    )


def flash_attention(q, k, v, causal=True, padding_mask=None):
    B, h, T, d = q.shape

    # raise error if causal is true and padding_mask is not None
    assert not (
        causal and padding_mask is not None
    ), "Causal and padding_mask not supported together"

    if padding_mask is not None:
        # torch.Size([1, 1, 1, T])
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.to(torch.bool)

    # apply flash attention
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    ):
        x = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    x = x.transpose(1, 2).reshape(B, T, d * h)
    return x


def inefficient_attention(q, k, v, causal=True):
    B, h, T, d = q.shape
    embed_dim = d * h

    attention = q.matmul(k.transpose(2, 3)) / math.sqrt(embed_dim // h)

    if causal:
        attn_mask = (
            get_causal_mask(q.shape[2])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(q.device)
            .to(q.dtype)
        )
        attention += attn_mask
    activation = torch.softmax(attention, dim=-1)
    x = (
        v.unsqueeze(2).repeat([1, 1, T, 1, 1])
        * activation.unsqueeze(-1).repeat([1, 1, 1, 1, 64])
    ).sum(dim=3)
    x = x.transpose(1, 2).reshape(B, T, d * h)
    return x


def create_sin_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions
    adim = torch.arange(half_dim, device=positions.device, dtype=positions.dtype).view(
        1, 1, -1
    )
    max_period_tensor = torch.full(
        [], 10_000, device=positions.device
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, dim=1024, num_heads=16, ff_dim=4096, rotary_emb=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiheadAttention(
            embed_dim=dim,
            causal=True,  # big difference!
            cross_attention=False,
            num_heads=num_heads,
            rotary_emb=rotary_emb,
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = MultiheadAttention(
            embed_dim=dim,
            causal=False,
            cross_attention=True,
            num_heads=num_heads,
            rotary_emb=rotary_emb,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ff_dim, bias=False)
        self.linear2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x, memory, memory_key_padding_mask):
        x_ = self.norm1(x)
        x = x + self.self_attn(x_, x_, x_)[0]

        x_ = self.norm_cross(x)
        x = x + self.cross_attention(x_, memory, memory, memory_key_padding_mask)[0]

        x_ = self.norm2(x)
        x = x + self.linear2(F.gelu(self.linear1(x_)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 24,
        ff_dim: int = 4096,
        heads: int = 16,
        rotary_emb: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerLayer(dim, heads, ff_dim, rotary_emb) for _ in range(depth)]
        )
        self._rotary_emb = rotary_emb

    def pos_embed(self, x):
        B, T, dim = x.shape
        positions = (
            torch.arange(T, device=x.device).view(1, -1, 1).to(x.dtype).to(x.device)
        )
        x = x + create_sin_embedding(positions, dim)
        return x

    def forward(self, x, memory, memory_key_padding_mask):
        if not self._rotary_emb:
            x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, memory, memory_key_padding_mask)
        return x


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(
        self,
        codebook_count: int,
        codebook_size: int,
        max_seq_len: int,
        dim: int,
        depth: int,
        num_heads: int,
        memory_dim: int,
        rotary_emb: bool = False,
    ):
        super().__init__()
        self.num_codebooks = codebook_count

        # set up the embeddings (for each codebook, we have an embedding + 1 for padding)
        embeddings = {f"codebook {k}": codebook_size + 1 for k in range(codebook_count)}

        self.transformer = MultiInputTransformerWrapper(
            num_tokens=embeddings,
            max_seq_len=max_seq_len,  # add the number of codebooks to the max_seq_len (for delay pattern)
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=num_heads,
                attn_flash=True,
                rotary_pos_emb=rotary_emb,
                cross_attend=True,
            ),
        )

        # memory projection
        if memory_dim != dim:
            self.memory_proj = nn.Linear(memory_dim, dim)
        else:
            self.memory_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: torch.Tensor = None,
        memory: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            x (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            x_padding_mask (torch.tensor): a tensor that will mask the padding
            memory (torch.tensor): a tensor that will feed the cross attention of shape
                (batch_size, seq_len, dim)
            memory_key_padding_mask (torch.tensor): a tensor that will mask the memory
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """
        # mask: true when attend, false when not attend

        memory = memory.to(x.device)
        memory = self.memory_proj(memory)

        # create the input dict that contains the codebooks for each embds layer
        x_bis = {}
        for k in range(self.num_codebooks):
            x_bis[f"codebook {k}"] = x[:, k, :]

        out = self.transformer(
            x_bis,
            mask=x_padding_mask,
            context=memory,
            context_mask=memory_key_padding_mask,
        )  # returns a dict with keys "codebook 0", "codebook 1", ... and values the probabilities for each token

        # stack the codebooks predictions to get the logits of shape (batch_size, num_codebooks, length, vocab_size)
        stacked_out = torch.stack(
            [out[f"codebook {k}"] for k in range(self.num_codebooks)], dim=1
        )

        return stacked_out

    def load_pretrained(self, device, model="facebook/musicgen-small"):
        """Load musicgen to test the model"""

        path = hf_hub_download(repo_id=model, filename="state_dict.bin", cache_dir=None)
        _values = torch.load(path, map_location=device)
        state_dict = {
            k: v for k, v in _values["best_state"].items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict)
