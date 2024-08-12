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
        self.model.eval()

        self._device = device

    def compress(self, x):
        with torch.no_grad():
            pred = self.model.compress(x)

        return pred.codes

    def encode(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model.encode(x)[1]

    def decompress(self, z: Tensor) -> Tensor:
        dac_file = dac.DACFile(
            codes=z,
            chunk_length=72,
            original_length=0,
            input_db=torch.tensor([-16]),
            channels=1,
            sample_rate=self.model.sample_rate,
            padding=False,
            dac_version="1.0.0",
        )

        with torch.no_grad():
            return self.model.decompress(dac_file)

    def decode(self, z: Tensor) -> Tensor:
        z = z.to(self._device)
        with torch.no_grad():
            z = self.model.quantizer.from_codes(z)[0]
            return self.model.decode(z)

    def sample_rate(self):
        return self.model.sample_rate
