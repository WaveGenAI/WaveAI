""" 
DAC model for audio compression.
"""

import dac
import torch
from torch import Tensor

from .autoencoder import AutoEncoder


class DAC(AutoEncoder):
    """DAC model for audio compression."""

    def __init__(self, device: torch.device, *args, **kwargs):
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.model.to(device)

    def compress(self, x):
        with torch.no_grad():
            pred = self.model.compress(x)

        return pred.codes

    def encode(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.encode(x)

    def decompress(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.decode(z)

    def decode(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.decode(z)

    def sample_rate(self):
        return self.model.sample_rate
