"""
SemanticCodec model for audio compression.
"""

import os
import tempfile

import torch
from audiotools import AudioSignal
from semanticodec import SemantiCodec
from torch import Tensor

from .autoencoder import AutoEncoder


class SementicCodec(AutoEncoder):
    """SemanticCodec model for audio compression."""

    def __init__(
        self,
        token_rate=100,
        semantic_vocab_size=8192,
        sample_rate=44100,
        *args,
        **kwargs
    ):

        self.model = SemantiCodec(
            token_rate=token_rate, semantic_vocab_size=semantic_vocab_size
        )

        self._sample_rate = sample_rate

    def compress(self, x):
        self.encode(x)

    def encode(self, x: Tensor) -> Tensor:
        audio = AudioSignal(x.cpu(), sample_rate=self._sample_rate)

        with tempfile.TemporaryDirectory() as tmpdirname:
            audio_path = os.path.join(tmpdirname, "temp.wav")
            audio.write(audio_path)

            with torch.no_grad():
                return self.model.encode(audio_path).transpose(1, 2)

    def decompress(self, z: Tensor) -> Tensor:
        return self.decode(z)

    def decode(self, z: Tensor) -> Tensor:
        z = z.transpose(1, 2)
        z = z.cpu()

        with torch.no_grad():
            wav = self.model.decode(z)

        # convert numpy.ndarray to torch.Tensor
        wav = torch.from_numpy(wav)

        return wav

    def sample_rate(self):
        return 16000
