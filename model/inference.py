""" 
Code for inference with the model.
"""

import audio_autoencoder
import text_encoder
import torch
from audiotools import AudioSignal
from config import Config

from model import WaveAILightning


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = Config()

        if path is None:
            self.model = WaveAILightning()
            self.model = WaveAILightning.load_from_checkpoint(path)
        else:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=512)

        audio_codec_path = audio_autoencoder.utils.download(model_type="44khz")
        self.audio_codec = audio_autoencoder.DAC.load(audio_codec_path)

    def greedy_decoding(self, src_text: str):
        """Perform greedy decoding with the model.

        Args:
            src_text (str): the source text to generate the audio from
        """

        encoded_text = self.text_encoder([src_text]).to(self.device)

        seq = torch.zeros(
            1,
            self.config.num_codebooks,
            self.config.max_seq_length - self.config.num_codebooks + 1,
        ).to(self.device)
        seq = seq + self.config.pad_token_id

        for i in range(500):
            logits, mask = self.model(seq, encoded_text)
            max_prob_idx = logits.argmax(dim=-1)

            seq[..., i + 1] = max_prob_idx[..., i + 1]

            print(
                f"Step {i + 1} / {self.config.max_seq_length - self.config.num_codebooks + 1}",
                end="\r",
            )

        output_ids = self.model.apply_delay_pattern_mask(seq, mask)

        output_ids = output_ids[output_ids != self.config.pad_token_id].reshape(
            1, self.config.num_codebooks, -1
        )

        output_ids = output_ids[None, ...].squeeze(0).long().cpu()

        z = self.audio_codec.quantizer.from_codes(output_ids)[0]
        y = self.audio_codec.decode(z)

        y = AudioSignal(y.detach().numpy(), sample_rate=44100)
        y.write("output.wav")


model = WaveModelInference(
    "lightning_logs/version_247/checkpoints/epoch=13-step=1400.ckpt"
)
model.greedy_decoding(
    "Calm piano and sustained violin motif suitable for study or sleep."
)
