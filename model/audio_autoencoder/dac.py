""" 
Descript audio codec wrapper
"""

import dac
import torch
from audiotools import AudioSignal


class DAC:
    """
    DAC Wrapper
    """

    def __init__(self, model_type: str = "44khz"):
        """Initialize the DAC model.

        Args:
            model_type (str, optional): the type of module to use. Defaults to "44khz".
        """
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        self.model.eval()

    @torch.no_grad()
    def encode(self, audio: str | torch.Tensor) -> torch.Tensor:
        """Encode the audio signal.

        Args:
            audio (str | torch.Tensor): the audio signal to encode.

        Returns:
            torch.Tensor: the encoded audio signal.
        """
        signal = AudioSignal(audio)
        signal = signal.resample(self.model.sample_rate)
        signal.to(self.model.device)

        x = self.model.preprocess(signal.audio_data, signal.sample_rate)

        x = x.transpose(0, 1)
        _, c, _, _, _ = self.model.encode(x)
        return c

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """The decode function.

        Args:
            codes (torch.Tensor): the codes to decode.

        Returns:
            torch.Tensor: the decoded audio signal.
        """
        z = self.model.quantizer.from_codes(codes)[0]
        y = self.model.decode(z)
        y = y.squeeze(1)
        return y
