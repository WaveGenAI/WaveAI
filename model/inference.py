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

        input_ids = torch.zeros(1, self.config.num_codebooks, 1).to(self.device)
        input_ids = input_ids + self.config.pad_token_id

        input_ids, mask = self.model.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        steps = self.config.max_seq_length - self.config.num_codebooks

        for i in range(steps):
            logits = self.model(input_ids, encoded_text)

            next_token_logits = logits[:, :, -1]

            _, indices = torch.topk(next_token_logits, 2)

            next_tokens = indices[:, :, 0].unsqueeze(-1)

            input_ids = torch.cat((input_ids, next_tokens), dim=-1)

            input_ids = self.model.apply_delay_pattern_mask(input_ids, mask)

            print(f"Step {i + 1} / {steps}", end="\r")

        output_ids = input_ids[input_ids != self.config.pad_token_id].reshape(
            1, self.config.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...].squeeze(0)

        z = self.audio_codec.quantizer.from_codes(output_ids.cpu())[0]

        y = torch.tensor([])

        with torch.no_grad():
            for i in range(0, z.shape[2], 200):
                print(f"Decoding {i} / {z.shape[2]}", end="\r")
                z_bis = z[:, :, i : i + 200]

                y_bis = self.audio_codec.decode(z_bis)
                y = torch.cat((y, y_bis), dim=-1)

        y = AudioSignal(y.numpy(), sample_rate=44100)
        y.write("output.wav")


model = WaveModelInference(
    "lightning_logs/version_276/checkpoints/epoch=18-step=1900.ckpt"
)

text = """ 
Emotional pop vocals with kicks, snares, hi-hats, synth leads, bass, and keys; instrumental with medium tempo, groovy bass, steady rhythm, cymbals, tambourine, keyboard; passionate vocals with punchy kicks, snares, hi-hats, synth leads, bass, keys; groovy synth cover with kicks, snares, hi-hats, claps, bass, keys, stereo crackle.

Lyric: 
""".strip()

model.greedy_decoding(text)
