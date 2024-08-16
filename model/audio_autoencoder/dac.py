""" 
DAC model for audio compression.
"""

import dac
import torch
from audiotools import AudioSignal
from torch import Tensor

from .autoencoder import AutoEncoder


class DAC(AutoEncoder):
    """DAC model for audio compression."""

    def __init__(self, device: torch.device, model_type: str):
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.to(device)
        self.model.eval()

        self._device = device

    def compress(self, x: torch.Tensor, sample_rate: int):
        audio = AudioSignal(x.cpu(), sample_rate=sample_rate)
        with torch.no_grad():
            pred = self.model.compress(audio)

        return pred.codes

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

    def sample_rate(self):
        return self.model.sample_rate
