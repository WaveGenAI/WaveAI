"""
Encodec model for audio compression.
"""

import torch
from encodec import EncodecModel
from torch import Tensor

from .autoencoder import AutoEncoder


class Encodec(AutoEncoder):
    """Encodec model for audio compression."""

    def __init__(self, device: torch.device, bandwidth: float = 6.0):
        self.model = EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(bandwidth)

        self.model.to(device)
        self._device = device

    def compress(self, x: Tensor, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            encoded_frames = self.model.encode(x)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes

    def decompress(self, z: Tensor) -> Tensor:
        z = z.to(self._device)
        with torch.no_grad():
            return self.model.decode([[z, None]])

    def sample_rate(self):
        return self.model.sample_rate
