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

import model.text_encoder as text_encoder
from model.audio_autoencoder import SementicCodec as audio_autoencoder


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
        overwrite: bool = False,
        save_dir: str = "./.data",
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): Path to the directory containing the audio files.
            duration (int, optional): duration of an audio. Defaults to 30.
            mono (bool, optional): convert to mono. Defaults to True.
            sample_rate (int, optional): sample rate of the audio. Defaults to 44100.
            max_length (int, optional): max length of the prompt. Defaults to 512.
            prompt (bool, optional): whether to use prompt. Defaults to False.
            overwrite (bool, optional): whether to overwrite the existing data. Defaults to False.
        """

        super().__init__()

        self.preprocess_and_save(
            audio_dir,
            save_dir,
            duration,
            mono,
            sample_rate,
            max_length,
            prompt,
            overwrite,
        )

        self.filenames = glob.glob(save_dir + "/*.pkl")
        self._prompt = prompt

    def preprocess_and_save(
        self,
        audio_dir: str,
        save_dir: str,
        duration: int = 30,
        mono: bool = True,
        sample_rate: int = 44100,
        max_length: int = 512,
        prompt: bool = False,
        overwrite: bool = False,
    ):

        if not overwrite and os.path.exists(save_dir):
            return

        if os.path.exists(save_dir):
            # ask the user if they want to overwrite the existing data
            success = False
            while not success:
                response = input(
                    f"The directory {save_dir} already exists. Do you want to overwrite it? (y/n): "
                )
                if response.lower() == "y":
                    success = True
                elif response.lower() == "n":
                    return
                else:
                    print("Invalid response. Please enter 'y' or 'n'.")

            shutil.rmtree(save_dir)

        os.makedirs(save_dir)

        filenames = glob.glob(audio_dir + "/*.mp3")
        txt_filenames = glob.glob(audio_dir + "/*.txt")

        if prompt:
            # remove all file names that do not have a corresponding text file
            filenames = [
                file_name
                for file_name in filenames
                if file_name.replace(".mp3", ".txt") in txt_filenames
            ]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        audio_codec = audio_autoencoder(
            device=device, bandwidth=3.0, sample_rate=sample_rate
        )
        text_enc = (
            text_encoder.T5EncoderBaseModel(max_length=max_length).eval().to(device)
        )

        for param in text_enc.parameters():
            param.requires_grad = False

        progress = 0
        for audio_file in filenames:
            try:
                audio = AudioSignal(
                    audio_file, duration=duration if duration > 0 else None
                )
            except (EOFError, RuntimeError):
                print(f"Error processing {audio_file}")
                continue

            audio.resample(sample_rate)

            if mono:
                audio = audio.to_mono()

            if audio.shape[-1] >= (duration * sample_rate):
                start_idx = random.randint(0, audio.shape[-1] - duration * sample_rate)
                audio = audio[:, :, start_idx : start_idx + duration * sample_rate]

            discret_audio_repr = audio_codec.encode(audio.audio_data.to(device))
            discret_audio_repr = discret_audio_repr.transpose(0, -1).to(device)

            data = {"audio": discret_audio_repr.cpu()}

            if prompt:
                with open(audio_file.replace(".mp3", ".txt"), encoding="utf-8") as f:
                    prompt_text = f.read()

                # pad the text to the max length
                prompt_text = prompt_text[:max_length].ljust(max_length, " ")

                text_latent = text_enc([prompt_text])
                data["text"] = text_latent.cpu()

            save_path = os.path.join(
                save_dir, os.path.basename(audio_file).replace(".mp3", ".pkl")
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

        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple:
        """Fetches the waveform for the given index.

        Args:
            index (int): Index of the waveform to fetch.

        Returns:
            tuple: A tuple containing the discret representation of the audio, the padding mask and the latent representation of the text.
        """

        with open(self.filenames[index], "rb") as f:
            data = pickle.load(f)

        if not self._prompt:
            return data["audio"], None

        return data["audio"], data["text"]

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

        text_reprs = torch.stack(text_reprs, dim=0).squeeze(1)

        return audio_reprs, text_reprs
