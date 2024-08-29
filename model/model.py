""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn

# from .transformer import Transformer
from x_transformers import Decoder, MultiInputTransformerWrapper


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(
        self,
        codebook_count: int,
        codebook_size: int,
        dim: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()

        self.emb = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)]
        )

        # self.transformer = Transformer(dim=dim, depth=depth, heads=num_heads)

        embds = {k: codebook_size + 1 for k in range(codebook_count)}

        self.transformer = MultiInputTransformerWrapper(
            num_tokens=embds,
            max_seq_len=4096,
            return_only_embed=True,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=num_heads,
                attn_flash=True,
                rotary_pos_emb=True,
            ),
        )

        self.out_norm = nn.LayerNorm(dim)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)]
        )

        self.num_codebooks = codebook_count

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            x (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        x_bis = {}
        for k in range(self.num_codebooks):
            x_bis[k] = x[:, k, :]
        x = self.transformer(x_bis)
        # x = self.out_norm(x)
        x = torch.stack([linear(x) for linear in self.linears], dim=1)
        return x
