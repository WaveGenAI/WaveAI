import torch
from torch import nn
from x_transformers import Decoder, MultiInputTransformerWrapper


class REPA(nn.Module):
    """Project the input sequence to a fixed size representation
    that can be used to compute the cosine similarity with an embedding
    https://arxiv.org/pdf/2410.06940
    """

    def __init__(self, hidden_size=512, projector_dim=2048, z_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        B, T, D = x.shape

        # apply the mlp
        x = self.mlp(x.reshape(-1, D))
        x = x.reshape(B, T, -1)

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
        repa_pos: int = 3,
    ):
        super().__init__()
        self.num_codebooks = codebook_count

        # set up the embeddings (for each codebook, we have an embedding + 3 for padding, start and end token)
        embeddings = {f"codebook {k}": codebook_size + 3 for k in range(codebook_count)}

        self.transformer = MultiInputTransformerWrapper(
            num_tokens=embeddings,
            max_seq_len=max_seq_len,  # add the number of codebooks to the max_seq_len (for delay pattern)
            emb_dropout=0.1,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=num_heads,
                attn_flash=True,
                cross_attend=True,
                rotary_pos_emb=rotary_emb,
                use_rmsnorm=True,
                ff_swish=True,
                ff_glu=True,
                attn_dropout=0.1,
                ff_dropout=0.1,
                attn_kv_heads=num_heads
                // 4,  # for example, if num_heads=32, attn_kv_heads=8 like llama3
            ),
        )

        # memory projection
        if memory_dim != dim:
            self.memory_proj = nn.Linear(memory_dim, dim)
        else:
            self.memory_proj = nn.Identity()

        self.repa = REPA(hidden_size=dim)
        self._repa_pos = repa_pos

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

        out, intermediates = self.transformer(
            x_bis,
            mask=x_padding_mask,
            context=memory,
            context_mask=memory_key_padding_mask,
            return_intermediates=True,
        )  # returns a dict with keys "codebook 0", "codebook 1", ... and values the probabilities for each token

        # stack the codebooks predictions to get the logits of shape (batch_size, num_codebooks, length, vocab_size)
        stacked_out = torch.stack(
            [out[f"codebook {k}"] for k in range(self.num_codebooks)], dim=1
        )

        # get the intermediate representation
        intermediate_repr = self.repa(intermediates.hiddens[self._repa_pos])

        return stacked_out, intermediate_repr
