"""
Layer for positional encoding
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Layer that add the sinusoidal positionnal encoding"""

    def __init__(self, dim_model: int, max_seq_len: int):
        """Initialize the PositionalEncoding layer

        Args:
            dim_model (int): the model dimension
            max_seq_len (int): the maximum sequence length
        """

        super().__init__()

        pe = torch.zeros((1, max_seq_len, dim_model))

        pos = torch.arange(max_seq_len).unsqueeze(1)
        divid = 10_000 ** (torch.arange(0, dim_model, 2) / dim_model)

        pe[0, :, 0::2] = torch.sin(pos / divid)
        pe[0, :, 1::2] = torch.cos(pos / divid)

        self.__pe = pe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """A method that adds the positional encoding to the input tensor

        Args:
            inputs (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the input tensor with the positional encoding
        """

        return inputs + self.__pe[:, : inputs.shape[1], :].to(inputs.device)
