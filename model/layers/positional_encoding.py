"""
Layer for positional encoding
"""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional embedding layer for Transformer networks"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create a long enough P matrix
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :].to(x.device)
        return x
