"""
Module to load the audio dataset.
"""

import glob
import random

import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset

import model.audio_autoencoder as audio_autoencoder
import model.text_encoder as text_encoder


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self,
        audio_dir: str,
        duration: int = 30,
        mono: bool = True,
        sample_rate: int = 44100,
        max_length: int = 512,
        prompt: bool = False,
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): Path to the directory containing the audio files.
            duration (int, optional): duration of an audio. Defaults to 30.
            mono (bool, optional): convert to mono. Defaults to True.
            sample_rate (int, optional): sample rate of the audio. Defaults to 44100.
            max_length (int, optional): max length of the prompt. Defaults to 512.
            prompt (bool, optional): whether to use prompt. Defaults to False.
        """

        super().__init__()

        self.filenames = glob.glob(audio_dir + "/*.mp3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._prompt = prompt
        self._mono = mono
        self._duration = duration
        self._sample_rate = sample_rate

        audio_codec_path = audio_autoencoder.utils.download(model_type="44khz")

        with torch.no_grad():  # fix serialize error
            self.audio_codec = audio_autoencoder.DAC.load(audio_codec_path).eval()
        self.audio_codec.to(self.device)

        for param in self.audio_codec.parameters():
            param.requires_grad = False

        self.text_encoder = text_encoder.T5EncoderBaseModel(
            max_length=max_length
        ).eval()
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
            tuple: A tuple containing the discret representation of the audio, the padding mask and the latent representation of the text.
        """

        audio_file = self.filenames[index]
        audio = AudioSignal(
            audio_file,
            duration=self._duration if self._duration > 0 else None,
        )
        audio.resample(self._sample_rate)

        if self._mono:
            audio = audio.to_mono()

        # truncate the audio if it is longer than the duration
        if audio.shape[-1] >= (self._duration * self._sample_rate):
            # random crop
            start_idx = random.randint(
                0, audio.shape[-1] - self._duration * self._sample_rate
            )
            audio = audio[
                :, :, start_idx : start_idx + self._duration * self._sample_rate
            ]

        with torch.no_grad():
            discret_audio_repr = self.audio_codec.compress(audio)
            discret_audio_repr = discret_audio_repr.codes.to(self.device)

            discret_audio_repr = discret_audio_repr[
                ..., : random.randint(10, 3000)
            ].transpose(0, -1)

        if not self._prompt:
            return discret_audio_repr, None

        with open(audio_file.replace(".mp3", ".txt"), encoding="utf-8") as f:
            prompt = f.read()

        return discret_audio_repr, prompt

    def collate_fn(self, batch: list) -> tuple:
        """Collates the batch.

        Args:
            batch (list): List of tuples containing the audio and text representations.

        Returns:
            tuple: A tuple containing the audio and text latent representations.
        """

        audio_reprs, text_reprs = zip(*batch)

        audio_reprs = (
            torch.nn.utils.rnn.pad_sequence(
                audio_reprs, batch_first=True, padding_value=-100
            )
            .transpose(1, 2)
            .squeeze(-1)
        )

        if not self._prompt:
            return audio_reprs, None

        with torch.no_grad():
            text_latent = self.text_encoder(text_reprs)

        return audio_reprs, text_latent
