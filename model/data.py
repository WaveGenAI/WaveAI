""" 
Preprocess the audio data and save it to disk.
Code is adapted from https://github.com/huggingface/parler-tts/blob/main/training/run_parler_tts_training.py
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from model.audio_autoencoder import DAC

from .config import Config


class AudioPreprocessor:
    def __init__(self, config: Config, private_dir: str = "./.data"):
        self.config = config
        self.private_dir = private_dir
        os.makedirs(self.private_dir, exist_ok=True)

        # Initialize audio codec and feature extractor
        self.audio_codec = DAC()
        self.audio_codec.dac.to("cuda")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "parler-tts/dac_44khZ_8kbps", feature_size=2
        )

        # Load dataset
        self.dataset_audio = load_dataset("WaveGenAI/audio_processed", streaming=True)

    @dataclass
    class DataCollatorEncodecWithPadding:
        feature_extractor: AutoFeatureExtractor
        audio_column_name: str
        feature_extractor_input_name: Optional[str] = "input_values"
        max_length: Optional[int] = None
        padding: Optional[str] = "longest"

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, str]]]:

            audios = [feature[self.audio_column_name]["array"] for feature in features]
            len_audio = [len(audio) for audio in audios]
            if self.max_length is not None:
                audios = [audio[:, : self.max_length] for audio in audios]

            sampling_rate = self.feature_extractor.sampling_rate
            batch = self.feature_extractor(
                audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length,
            )
            batch["len_audio"] = torch.tensor(len_audio).unsqueeze(1)

            for feature in features:
                feature.pop(self.audio_column_name)

            return batch, features

    def _preprocess_and_save(self, amount: int = -1):
        # Initialize DataLoader with DataCollator
        encoder_data_collator = self.DataCollatorEncodecWithPadding(
            self.feature_extractor,
            audio_column_name="audio",
            max_length=self.config.data.duration * self.config.codec.sample_rate,
        )

        data_loader = DataLoader(
            self.dataset_audio["train"],
            batch_size=self.config.data.dataset_preprocess_batch_size,
            collate_fn=encoder_data_collator,
        )

        # Process and save the dataset in chunks
        batch_audio = []
        for i, batch in tqdm(enumerate(data_loader)):
            # Stop if the amount of samples is reached
            if (
                amount != -1
                and (i * self.config.data.dataset_preprocess_batch_size) >= amount
            ):
                break

            batch, features = batch

            with torch.no_grad():
                _, s = self.audio_codec.encode(batch["input_values"])
            print(s.shape)
            for row in range(s.shape[0]):
                out = {"audio": s[row]}

                # Define the rest of the columns
                for key in features[0].keys():
                    out[key] = features[row][key]

                batch_audio.append(out)

            # if the batch is too big, save it to disk
            if len(batch_audio) > 50:
                continue

            self._save_to_disk(batch_audio)
            batch_audio = []

        # Save the last batch
        if len(batch_audio) > 0:
            self._save_to_disk(batch_audio)

        # Concatenate and return the final dataset
        return self._load_and_concatenate_datasets()

    def _save_to_disk(self, batch_audio):
        dataset = Dataset.from_list(batch_audio)
        dataset.save_to_disk(
            os.path.join(self.private_dir, str(len(os.listdir(self.private_dir))))
        )

    def _load_and_concatenate_datasets(self):
        datasets = [
            load_from_disk(os.path.join(self.private_dir, checkpoint))
            for checkpoint in os.listdir(self.private_dir)
        ]
        final_dataset = concatenate_datasets(datasets, axis=0).with_format("torch")
        final_dataset.save_to_disk(os.path.join(self.private_dir, "encoded_dataset"))
        return final_dataset

    def get_dataset(self, amount: int = -1):
        if not os.path.exists(os.path.join(self.private_dir, "encoded_dataset")):
            data = self._preprocess_and_save(amount)
        else:
            data = load_from_disk(os.path.join(self.private_dir, "encoded_dataset"))

        # select the first `amount` samples
        data = data.select(range(amount))
        return data
