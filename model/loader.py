"""
Module to load the audio dataset.
"""

import glob

import audio_autoencoder
import text_encoder
import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self,
        audio_dir: str,
        max_duration: int = 30,
        mono: bool = True,
        sample_rate: int = 44100,
        max_length: int = 512,
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): Path to the directory containing the audio files.
            max_duration (int, optional): max duration of an audio. Defaults to 30.
            mono (bool, optional): convert to mono. Defaults to True.
            sample_rate (int, optional): sample rate of the audio. Defaults to 44100.
            max_length (int, optional): max length of the prompt. Defaults to 512.
        """

        super().__init__()

        self.filenames = glob.glob(audio_dir + "/*.mp3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._mono = mono
        self._cut_length = max_duration
        self._sample_rate = sample_rate

        audio_codec_path = audio_autoencoder.utils.download(model_type="44khz")
        self.audio_codec = audio_autoencoder.DAC.load(audio_codec_path)
        self.audio_codec.to(self.device)

        for param in self.audio_codec.parameters():
            param.requires_grad = False

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=max_length)
        self.text_encoder.to(self.device)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def __len__(self) -> int:
        """Returns the number of waveforms in the dataset.

        Returns:
            int: Number of waveforms in the dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple:
        """Fetches the waveform for the given index.

        Args:
            index (int): Index of the waveform to fetch.

        Returns:
            tuple: A tuple containing the discret representation of the audio and the latent representation of the text.
        """

        audio_file = self.filenames[index]
        audio = AudioSignal(
            audio_file,
            duration=self._cut_length if self._cut_length > 0 else None,
            sample_rate=44100,
        )

        if self._mono:
            audio = audio.to_mono()

        audio.zero_pad_to(0, self._cut_length * self._sample_rate - audio.signal_length)

        discret_audio_repr = self.audio_codec.compress(audio)
        discret_audio_repr = discret_audio_repr.codes.to(self.device)

        with open(audio_file.replace(".mp3", ".txt"), encoding="utf-8") as f:
            prompt = f.read()

        return (discret_audio_repr, prompt)

    def collate_fn(self, batch: list) -> tuple:
        """Collates the batch.

        Args:
            batch (list): List of tuples containing the audio and text representations.

        Returns:
            tuple: A tuple containing the audio and text latent representations.
        """

        audio_reprs, text_reprs = zip(*batch)

        text_latent = self.text_encoder(text_reprs)

        audio_reprs = torch.stack(audio_reprs).squeeze(1)

        return audio_reprs, text_latent
