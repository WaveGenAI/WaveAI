""" 
Filter audio data using CLAP model.
"""

import argparse
import logging
import os

import laion_clap
import numpy as np
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("--audio_dir", type=str, required=True)
argparser.add_argument("--threshold", type=float, required=False, default=0.1)
argparser.add_argument("--delete", action="store_true")
args = argparser.parse_args()


logger = logging.getLogger(__name__)


class FilterClap:
    """
    FilterClap class to filter audio data using CLAP model.
    """

    def __init__(self) -> None:
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()

    def _get_text_embedding(
        self, text_data: str, use_tensor: bool = True
    ) -> torch.Tensor:
        """Calculate text embedding from text data.

        Args:
            text_data (str): text data to be embedded.
            use_tensor (bool, optional): convert to tensor. Defaults to True.

        Returns:
            torch.Tensor: text embedding.
        """

        text_embed = self.model.get_text_embedding([text_data], use_tensor=use_tensor)
        return text_embed

    def _get_audio_embedding_from_filelist(
        self, path_audio: str, use_tensor: bool = True
    ) -> torch.Tensor:
        """Calculate audio embedding from audio file.

        Args:
            path_audio (str): path to audio file.
            use_tensor (bool, optional): convert to tensor. Defaults to True.

        Returns:
            torch.Tensor: audio embedding.
        """

        audio_embed = self.model.get_audio_embedding_from_filelist(
            x=[path_audio], use_tensor=use_tensor
        )
        return audio_embed

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> int:
        """Calculate cosine similarity between two vectors.

        Args:
            x (torch.Tensor): input vector.
            y (torch.Tensor): input vector.

        Returns:
            int: the cosine similarity between two vectors.
        """

        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def sim_prob(self, text_data: str, path_audio: str) -> int:
        """Calculate similarity probability between text and audio.

        Args:
            text_data (str): text data to be embedded.
            path_audio (str): path to audio

        Returns:
            int: similarity probability between text and audio.
        """

        text_embed = self._get_text_embedding(text_data)
        audio_embed = self._get_audio_embedding_from_filelist(path_audio)

        text_embed = text_embed.cpu().detach().numpy()[0]
        audio_embed = audio_embed.cpu().detach().numpy()[0]
        sim = self._cosine_similarity(text_embed, audio_embed)
        return sim


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_clap = FilterClap()

    for file in os.listdir(args.audio_dir):
        if not file.endswith(".txt"):
            continue

        base_name = os.path.splitext(file)[0]
        text_file = os.path.join(args.audio_dir, file)
        audio_file = os.path.join(args.audio_dir, base_name + ".mp3")

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        text = text.split("Lyric:")[0]

        sim_prob = filter_clap.sim_prob(text_data=text.strip(), path_audio=audio_file)

        if sim_prob < args.threshold:
            logger.info("File: %s is not similar to %s", audio_file, text_file)

            if args.delete:
                os.remove(audio_file)
                os.remove(text_file)
