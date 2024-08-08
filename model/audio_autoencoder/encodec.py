"""
Encodec model for audio compression.
"""

import torch
from encodec import EncodecModel
from torch import Tensor

from .autoencoder import AutoEncoder


class Encodec(AutoEncoder):
    """Encodec model for audio compression."""

    def __init__(self, device: torch.device, bandwidth: float = 6.0, *args, **kwargs):
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)

        self.model.to(device)

    def compress(self, x):
        raise NotImplementedError

    def encode(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            encoded_frames = self.model.encode(x)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes

    def decompress(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.decode([[z, None]])
