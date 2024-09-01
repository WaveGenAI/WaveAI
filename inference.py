""" 
Code for inference with the model.
"""

import torch
from audiotools import AudioSignal
from transformers import AutoTokenizer

import model.text_encoder as text_encoder
from model.audio_autoencoder import DAC as audio_autoencoder
from model.config import Config
from model.generation import Generation
from model.lightning_model import WaveAILightning

config = Config()


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

        self.text_encoder = text_encoder.T5EncoderBaseModel()

        self.audio_codec = audio_autoencoder()

        self._tok = AutoTokenizer.from_pretrained(config.model.tokenizer)

        self.num_codebooks = config.model.num_codebooks
        if config.model.stereo:
            self.num_codebooks = self.num_codebooks * 2

        self.generation = Generation(
            self.model,
            self.num_codebooks,
            config.model.pad_token_id,
            config.model.stereo,
        )

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

        prompt_embd, prompts_masks = self.text_encoder([src_text])
        # lyric_ids = self._tok(lyrics, return_tensors="pt").input_ids

        prompt_embd = prompt_embd.to(self.device)
        prompts_masks = prompts_masks.to(self.device)

        output_ids = self.generation.sampling(prompt_embd, prompts_masks)

        with torch.no_grad():
            y = self.audio_codec.decode(output_ids)

        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.model.sample_rate)
        y.write("output.wav")


model = WaveModelInference()

prompt = """ 
subject: baltimore club,bass music,electronic,experimental,jersey club,juke,jungle,vaporwave,future funk,screwgaze,television,vaporwave,vhs,philadelphia, title: Touch The Ground, album: Culture Vulture, genre:
""".strip()

lyric = """
I'm a little teapot
""".strip()

model.sampling(prompt, lyric)
