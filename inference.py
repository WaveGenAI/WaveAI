""" 
Code for inference with the model.
"""

import torch
from audiotools import AudioSignal

import model.text_encoder as text_encoder
from model.audio_autoencoder import Encodec as audio_autoencoder
from model.generation import Generation
from model.lightning_model import WaveAILightning


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = WaveAILightning()
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model.to(self.device)
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=512)

        self.audio_codec = audio_autoencoder(device=self.device)

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

        if isinstance(src_text, str):
            encoded_text = self.text_encoder([src_text])
        else:
            encoded_text = src_text

        encoded_text = encoded_text.to(self.device)

        input_ids = input_ids[:, :, : self.model.config.max_seq_length]

        output_ids = self.generation.sampling(None, input_ids)

        output_ids = output_ids[output_ids != self.model.config.pad_token_id].reshape(
            1, self.model.config.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...].squeeze(0)

        y = torch.tensor([], device=self.device)

        with torch.no_grad():
            for i in range(0, output_ids.shape[-1], 200):
                print(f"Decoding {i} / {output_ids.shape[2]}", end="\r")
                z_bis = output_ids[:, :, i : i + 200]

                y_bis = self.audio_codec.decode(z_bis)
                y = torch.cat((y, y_bis), dim=-1)

        y = AudioSignal(y.cpu().numpy(), sample_rate=44100)
        y.write("output.wav")


if __name__ == "__main__":
    model = WaveModelInference("epoch=5-step=1038.ckpt")

    text = """ 
    bass guitar with drums and piano 
    """.strip()

    from torch.utils.data import DataLoader

    from model.loader import SynthDataset

    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")
    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn
    )

    first_batch = next(iter(train_loader))

    model.sampling(text, first_batch[0][..., :500])
