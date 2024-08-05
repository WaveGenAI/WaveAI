""" 
Code for inference with the model.
"""

import audio_autoencoder
import text_encoder
import torch
from audiotools import AudioSignal
from config import Config
from generation import Generation

from model import WaveAILightning


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

        self.generation = Generation(self.model)

    def sampling(
        self,
        src_text: str | torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ):
        """Sampling with the model.

        Args:
            src_text (str): the source text to generate the audio from
            input_ids (torch.Tensor, optional): the input ids to start the generation from. Defaults to None.
        """

        if input_ids is not None:
            z = self.audio_codec.quantizer.from_codes(input_ids.cpu())[0]

            y = torch.tensor([])

            with torch.no_grad():
                for i in range(0, z.shape[2], 200):
                    z_bis = z[:, :, i : i + 200]

                    y_bis = self.audio_codec.decode(z_bis)
                    y = torch.cat((y, y_bis), dim=-1)

            y = AudioSignal(y.numpy(), sample_rate=44100)
            y.write("output_bis.wav")

        if isinstance(src_text, str):
            encoded_text = self.text_encoder([src_text])
        else:
            encoded_text = src_text

        encoded_text = encoded_text.to(self.device)

        input_ids = input_ids[
            :, :, : self.config.max_seq_length - self.config.num_codebooks
        ]

        _, mask = self.model.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        output_ids = self.generation.sampling(mask, None, input_ids)
        output_ids = self.model.apply_delay_pattern_mask(output_ids, mask)

        output_ids = output_ids[output_ids != self.config.pad_token_id].reshape(
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


if __name__ == "__main__":
    model = WaveModelInference("WAVEAI/ha388fe4/checkpoints/epoch=0-step=173.ckpt")

    text = """ 
    bass guitar with drums and piano 
    """.strip()

    from loader import SynthDataset
    from torch.utils.data import DataLoader

    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")
    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn
    )

    first_batch = next(iter(train_loader))

    model.sampling(text, first_batch[0][..., :500])
