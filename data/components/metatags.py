"""
Add metatags to a song with CLAP model.
Code adapted from https://github.com/lyramakesmusic/clap-interrogator/blob/main/clap_interrogator.py
"""

import os
from typing import List

import librosa
import torch
from transformers import ClapModel, ClapProcessor


class CLAPTagger:
    """
    A class to tag a song with the CLAP model.
    """

    def __init__(self, prob_threshold=0.5, model: str = "laion/clap-htsat-unfused"):
        self.processor = ClapProcessor.from_pretrained(model)
        self.model = ClapModel.from_pretrained(model)

        self._tags = {}
        self._load_tags()
        self.prob_threshold = prob_threshold

    def _load_tags(self):
        """
        Load tags from the tags directory.
        """

        current_dir = os.path.dirname(os.path.realpath(__file__))
        for file in os.listdir(os.path.join(current_dir, "tags")):
            if file.endswith(".txt"):
                with open(
                    os.path.join(current_dir, "tags", file), "r", encoding="utf-8"
                ) as f:
                    tags = f.read().splitlines()

                    self._tags[file[:-4]] = tags

    def tag(self, audio_path: str) -> List[str]:
        """Tag an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            List[str]: List of tags.
        """

        audio, sr = librosa.load(audio_path, sr=48000)

        list_tags = []
        for _, tag_lst in self._tags.items():
            features = self.processor(
                text=tag_lst,
                audios=[audio],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )  # preprocess the data

            with torch.no_grad():
                outputs = self.model(
                    **features
                )  # CLAP will return the logits for each audio
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1)  # convert logits to probabilities

            top_probs, top_indices = probs.topk(
                10, dim=1
            )  # get the top 10 probabilities

            for prob, idx in zip(top_probs[0], top_indices[0]):
                if prob < self.prob_threshold:
                    continue

                list_tags.append(tag_lst[idx])

        return list_tags
