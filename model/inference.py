""" 
Code for inference with the model.
"""

import audio_autoencoder
import text_encoder
import torch
from audiotools import AudioSignal
from config import Config

from model import WaveAILightning
import time


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = Config()

        self.model = WaveAILightning()
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model.to(self.device)
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

        input_ids = torch.zeros(1, self.config.num_codebooks, 1)
        input_ids = input_ids + self.config.pad_token_id

        # delay pattern used by Musicgen
        input_ids, mask = self.model.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        input_ids = input_ids.to(self.device)
        all_tokens = [input_ids]

        for i in range(self.config.max_seq_length):
            inputs = torch.cat(all_tokens, dim=-1)

            logits, _ = self.model(inputs, encoded_text)
            max_prob_idx = logits.argmax(dim=-1)

            output_ids = max_prob_idx[..., -1].unsqueeze(-1)

            all_tokens.append(output_ids)

            print(f"Step {i + 1} / {self.config.max_seq_length}", end="\r")

        output_ids = torch.cat(all_tokens, dim=-1)
        output_ids = self.model.apply_delay_pattern_mask(output_ids.cpu(), mask)

        output_ids = output_ids[output_ids != self.config.pad_token_id].reshape(
            1, self.config.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...].squeeze(0)

        z = self.audio_codec.quantizer.from_codes(output_ids.cpu())[0]
        y = torch.tensor([])

        for i in range(0, z.shape[2], 200):
            print(f"Decoding {i} / {z.shape[2]}", end="\r")
            z_bis = z[:, :, i : i + 200]

            y_bis = self.audio_codec.decode(z_bis)
            y = torch.cat((y, y_bis), dim=-1)

        y = AudioSignal(y.detach().numpy(), sample_rate=44100)
        y.write("output.wav")


model = WaveModelInference(
    "lightning_logs/version_246/checkpoints/epoch=3-step=400.ckpt"
)
model.greedy_decoding(
    "Calm piano and sustained violin motif suitable for study or sleep."
)
