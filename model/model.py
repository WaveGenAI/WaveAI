""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn

from .transformer import Transformer


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(
        self,
        codebook_count: int,
        codebook_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        memory_dim: int,
    ):
        super().__init__()

        self.emb = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)]
        )

        self.transformer = Transformer(dim=dim, depth=depth, heads=num_heads)

        self.out_norm = nn.LayerNorm(dim)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)]
        )

        self.num_codebooks = codebook_count

        if memory_dim != dim:
            self.memory_proj = nn.Linear(memory_dim, dim)
        else:
            self.memory_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            x (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            memory (torch.tensor): a tensor that will fee the cross attention of shape
                (batch_size, seq_len, dim)
            memory_key_padding_mask (torch.tensor): a tensor that will mask the memory
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        x = sum([emb(x[:, i, :]) for i, emb in enumerate(self.emb)])

        memory = self.memory_proj(memory).to(x.device)
        x = self.transformer(x, memory, memory_key_padding_mask)
        x = self.out_norm(x)
        x = torch.stack([linear(x) for linear in self.linears], dim=1)
        return x
