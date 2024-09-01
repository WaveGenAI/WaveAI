import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, causal: bool, cross_attention: bool
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.cross_attention = cross_attention

        self.in_proj_weight = nn.Parameter(torch.rand((embed_dim * 3, embed_dim)))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        rotary_dim = (
            embed_dim // num_heads
        ) // 2  # rotary embedding dim = dim of each head / 2
        self.rotary_emb = RotaryEmbedding(dim=rotary_dim)

    def forward(self, query, key, value, padding_mask=None):
        if not self.cross_attention:
            key = query
            value = query

        q = F.linear(query, self.in_proj_weight[: self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim : 2 * self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2 * self.embed_dim :])

        q, k, v = [
            x.reshape(
                x.shape[0], self.num_heads, x.shape[1], x.shape[2] // self.num_heads
            )
            for x in [q, k, v]
        ]

        B, h, T, d = q.shape

        # apply rotary embedding to queries and keys (if not cross-attention)
        q = self.rotary_emb.rotate_queries_or_keys(q)

        if not self.cross_attention:
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # create causal mask
        temp_mask = torch.ones(
            B, h, q.size(-2), k.size(-2), dtype=torch.bool, device=q.device
        )

        if self.causal:
            temp_mask = temp_mask.tril(diagonal=0)

        if padding_mask is not None:
            # be sure that padding mask is boolean
            padding_mask = padding_mask.bool()

            # temp_mask is [B, h, QT, KT] and padding_mask is [B, KT] so expand padding mask to [B, 1, 1, KT]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

            temp_mask = temp_mask & padding_mask

        # apply flash attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=temp_mask)

        x = x.reshape(B, T, self.embed_dim)
        x = self.out_proj(x)
        return x, None


def create_sin_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=positions.device, dtype=positions.dtype).view(
        1, 1, -1
    )
    max_period_tensor = torch.full([], 10000, device=positions.device)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, ff_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiheadAttention(
            embed_dim=dim, causal=True, cross_attention=False, num_heads=heads
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = MultiheadAttention(
            embed_dim=dim, causal=False, cross_attention=True, num_heads=heads
        )
        self.dropout_cross = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ff_dim, bias=False)
        self.linear2 = nn.Linear(ff_dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None):
        x_ = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_, x_, x_)[0])

        x_ = self.norm_cross(x)
        x = x + self.dropout_cross(
            self.cross_attention(x_, memory, memory, memory_key_padding_mask)[0]
        )

        x_ = self.norm2(x)
        x = x + self.dropout2(self.linear2(F.gelu(self.linear1(x_))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 24,
        ff_dim: int = 4096,
        heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(dim, ff_dim, heads, dropout) for _ in range(depth)]
        )

    def forward(self, x, memory, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x
