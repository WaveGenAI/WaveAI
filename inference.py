""" 
Code for inference with the model.
"""

import torch
from audiotools import AudioSignal
from transformers import AutoTokenizer

import model.text_encoder as text_encoder
from model.config import Config
from model.generation import Generation
from model.lightning_model import WaveAILightning

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

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = WaveAILightning()

        path = config.inference.checkpoint_path
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)
            print(f"Loaded model from {path}")

        self.model = self.model.model.to(self.device)
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(
            max_length=config.data.max_prompt_length
        )

        if config.codec.name in config.__dict__.keys():
            self.audio_codec = audio_autoencoder(
                device=self.device, **config.__dict__[config.codec.name].__dict__
            )
        else:
            self.audio_codec = audio_autoencoder(device=self.device)

        self._tok = AutoTokenizer.from_pretrained(config.model.tokenizer)

        self.generation = Generation(self.model, device=self.device)

    def sampling(
        self,
        src_text: str,
        lyrics: str,
    ):
        """Sampling with the model.

        Args:
            src_text (str): the source text to generate the audio from
            input_ids (torch.Tensor, optional): the input ids to start the generation from. Defaults to None.
        """

        prompt_embd = self.text_encoder([src_text])
        lyric_ids = self._tok(lyrics, return_tensors="pt").input_ids

        prompt_embd = prompt_embd.to(self.device)
        lyric_ids = lyric_ids.to(self.device)

        output_ids = self.generation.sampling(
            prompt_embd, lyric_ids, top_k=self.model.config.inference.top_k
        )
        output_ids = output_ids[:, :, :]

        with torch.no_grad():
            y = self.audio_codec.decompress(output_ids)

        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.sample_rate())
        y.write("output.wav")


model = WaveModelInference()

prompt = """ 
subject: baltimore club,bass music,electronic,experimental,jersey club,juke,jungle,vaporwave,future funk,screwgaze,television,vaporwave,vhs,philadelphia, title: Touch The Ground, album: Culture Vulture, genre:
""".strip()

lyric = """
I'm a little teapot
""".strip()

model.sampling(prompt, lyric)
