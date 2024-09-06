""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

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
        rotary_emb: bool = False,
    ):
        super().__init__()

        self.emb = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, dim) for _ in range(codebook_count)]
        )

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=num_heads, rotary_emb=rotary_emb
        )

        self.out_norm = nn.LayerNorm(dim)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, codebook_size, bias=False) for _ in range(codebook_count)]
        )

        self.num_codebooks = codebook_count

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

        memory = memory.to(x.device)
        x = self.transformer(x, memory, memory_key_padding_mask)
        x = self.out_norm(x)
        x = torch.stack([linear(x) for linear in self.linears], dim=1)
        return x

    def load_pretrained(self, device, model="facebook/musicgen-small"):
        path = hf_hub_download(repo_id=model, filename="state_dict.bin", cache_dir=None)
        _values = torch.load(path, map_location=device)
        state_dict = {
            k: v for k, v in _values["best_state"].items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict)
