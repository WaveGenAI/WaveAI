"""
Module to load the audio dataset.
"""

import glob
import os
import pickle
import random
import shutil

import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import model.text_encoder as text_encoder
from model.config import Config

config = Config()
codec = config.codec.name

if codec == "DAC":
    from model.audio_autoencoder import DAC as audio_autoencoder
elif codec == "Encodec":
    from model.audio_autoencoder import Encodec as audio_autoencoder
elif codec == "SementicCodec":
    from model.audio_autoencoder import SementicCodec as audio_autoencoder
else:
    raise ValueError("Invalid codec")


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self,
        overwrite: bool = False,
    ):
        """Initializes the dataset.

        Args:
            overwrite (bool, optional): whether to overwrite the existing data. Defaults to False.
        """

        super().__init__()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_enc = (
            text_encoder.T5EncoderBaseModel(max_length=config.data.max_prompt_length)
            .eval()
            .to(self._device)
        )

        for param in self.text_enc.parameters():
            param.requires_grad = False

        self._tok = AutoTokenizer.from_pretrained(config.model.tokenizer)

        self._audio_dir = config.data.audio_dir
        self._save_dir = config.data.save_dir
        self._duration = config.data.duration
        self._sample_rate = config.codec.sample_rate
        self._overwrite = overwrite

        self.audio_codec = audio_autoencoder(
            device=self._device, **config.__dict__[config.codec.name].__dict__
        )

        self.preprocess_and_save()

        self._filenames = glob.glob(self._save_dir + "/*.pkl")

    def preprocess_and_save(self):
        """Preprocesses the audio files and saves them to the disk."""

        if not self._overwrite and os.path.exists(self._save_dir):
            return

        if os.path.exists(self._save_dir):
            # ask the user if they want to overwrite the existing data
            success = False
            while not success:
                response = input(
                    f"The directory {self._save_dir} already exists. Do you want to overwrite it? (y/n): "
                )
                if response.lower() == "y":
                    success = True
                elif response.lower() == "n":
                    return
                else:
                    print("Invalid response. Please enter 'y' or 'n'.")

            shutil.rmtree(self._save_dir)

        os.makedirs(self._save_dir)

        filenames = glob.glob(self._audio_dir + "/*.mp3")

        progress = 0
        for audio_file in filenames:
            if not os.path.exists(audio_file.replace(".mp3", "_descr.txt")):
                continue

            if not os.path.exists(audio_file.replace(".mp3", "_transcript.txt")):
                continue

            try:
                audio = AudioSignal(
                    audio_file, duration=self._duration if self._duration > 0 else None
                )
            except (EOFError, RuntimeError):
                print(f"Error processing {audio_file}")
                continue

            audio.resample(self._sample_rate)

            if audio.shape[-1] >= (self._duration * self._sample_rate):
                start_idx = random.randint(
                    0, audio.shape[-1] - self._duration * self._sample_rate
                )
                audio = audio[
                    :, :, start_idx : start_idx + self._duration * self._sample_rate
                ]

            codes = []

            for channel in range(audio.num_channels):
                discret_audio_repr = self.audio_codec.compress(
                    audio.audio_data[:, channel, :].unsqueeze(1).to(self._device),
                    self._sample_rate,
                ).transpose(0, -1)
                codes.append(discret_audio_repr)

            for idx, code in enumerate(codes):
                discret_audio_repr = code.cpu()
                data = {"audio": discret_audio_repr.cpu()}

                with open(
                    audio_file.replace(".mp3", "_descr.txt"), encoding="utf-8"
                ) as f:
                    data["prompt"] = f.read().strip()

                with open(
                    audio_file.replace(".mp3", "_transcript.txt"), encoding="utf-8"
                ) as f:
                    data["transcript"] = f.read().strip()

                save_path = os.path.join(
                    self._save_dir,
                    f"({idx})" + os.path.basename(audio_file).replace(".mp3", ".pkl"),
                )

                with open(save_path, "wb") as f:
                    pickle.dump(data, f)

            progress += 1
            print(f"Progress: {progress}/{len(filenames)}", end="\r")

    def __len__(self) -> int:
        """Returns the number of waveforms in the dataset.

        Returns:
            int: Number of waveforms in the dataset.
        """

        return len(self._filenames)

    def __getitem__(self, index: int) -> tuple:
        """Fetches the waveform for the given index.

        Args:
            index (int): Index of the waveform to fetch.

        Returns:
            tuple: A tuple containing the discret representation of the audio, the padding mask and the latent representation of the text.
        """

        with open(self._filenames[index], "rb") as f:
            data = pickle.load(f)

        return data

    def collate_fn(self, batch: list) -> tuple:
        """Collates the batch.

        Args:
            batch (list): List of tuples containing the audio and text representations.

        Returns:
            tuple: A tuple containing the audio and text latent representations of prompt and lyrics.
        """
        audio = [torch.tensor(item["audio"]) for item in batch]
        prompt = [item["prompt"] for item in batch]
        lyrics = [item["transcript"] for item in batch]

        codes = (
            torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=-100)
            .transpose(1, 2)
            .squeeze(-1)
        )

        # tokenize the lyrics
        lyrics_ids = self._tok(
            lyrics,
            padding=True,
            truncation=True,
            max_length=config.data.max_lyrics_length,
            return_tensors="pt",
        ).input_ids

        prompts_embd = self.text_enc(prompt)

        return codes, prompts_embd, lyrics_ids
