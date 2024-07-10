""" 
Train script for WaveAI
"""

from typing import Any
import lightning as L
import audio_autoencoder
import text_encoder


class WaveAI(L.LightningModule):
    def __init__(self, audio_codec, encoder) -> None:
        super().__init__()

        self.audio_codec = audio_codec
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


if __name__ == "__main__":
    audio_codec_path = audio_autoencoder.utils.download(model_type="44khz")
    audio_codec = audio_autoencoder.DAC.load(audio_codec_path)

    encoder = text_encoder.T5EncoderBaseModel()

    wave_ai = WaveAI(audio_codec, encoder)
