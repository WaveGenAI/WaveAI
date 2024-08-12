""" 
Code for inference with the model.
"""

import torch
from audiotools import AudioSignal
from torch.utils.data import DataLoader

import model.text_encoder as text_encoder
from model.config import Config
from model.generation import Generation
from model.lightning_model import WaveAILightning
from model.loader import SynthDataset

config = Config()

if config.codec.name == "DAC":
    from model.audio_autoencoder import DAC as audio_autoencoder
elif config.codec.name == "Encodec":
    from model.audio_autoencoder import Encodec as audio_autoencoder
elif config.codec.name == "SementicCodec":
    from model.audio_autoencoder import SementicCodec as audio_autoencoder
else:
    raise ValueError("Invalid codec")


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = WaveAILightning()
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model.to(self.device)
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=512)

        if config.codec.name in config.__dict__.keys():
            self.audio_codec = audio_autoencoder(
                device=self.device, **config.__dict__[config.codec.name].__dict__
            )
        else:
            self.audio_codec = audio_autoencoder(device=self.device)

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

        input_ids = input_ids[:, :, : self.model.config.model.max_seq_length]

        output_ids = self.generation.sampling(None, input_ids, top_k=50)
        output_ids = output_ids[:, :, :]

        with torch.no_grad():
            y = self.audio_codec.decompress(output_ids)

        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.sample_rate())
        y.write("output.wav")


model = WaveModelInference(config.inference.checkpoint_path)

text = """ 
bass guitar with drums and piano 
""".strip()

codec_args = None
if config.codec.name in config.__dict__.keys():
    codec_args = config.__dict__[config.codec.name]

dataset = SynthDataset(
    audio_dir=config.data.audio_dir,
    save_dir=config.data.save_dir,
    duration=config.data.duration,
    prompt=config.data.prompt,
    overwrite=False,
    codec=config.codec.name,
    config_codec=codec_args,
)
train_loader = DataLoader(
    dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn
)

first_batch = next(iter(train_loader))

model.sampling(text, first_batch[0][..., :300])
