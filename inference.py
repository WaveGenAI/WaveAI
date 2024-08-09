""" 
Code for inference with the model.
"""

import torch
from audiotools import AudioSignal

import model.text_encoder as text_encoder
from model.audio_autoencoder import SementicCodec as audio_autoencoder
from model.generation import Generation
from model.lightning_model import WaveAILightning


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = WaveAILightning()
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model.to(self.device)
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=512)
        self.audio_codec = audio_autoencoder(bandwidth=3.0, device=self.device)

        self.generation = Generation(self.model, device=self.device)

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

        output_ids = self.generation.sampling(None, input_ids, top_k=100)

        with torch.no_grad():
            try:
                y = self.audio_codec.decompress(output_ids)
            except NotImplementedError:
                y = self.audio_codec.decode(output_ids)

        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.sample_rate())
        y.write("output.wav")


if __name__ == "__main__":
    model = WaveModelInference("WAVEAI/an8y8dt1/checkpoints/epoch=9-step=970.ckpt")

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

    model.sampling(text, first_batch[0][..., :200])
