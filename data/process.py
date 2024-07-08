"""
Dataset Builder
"""

import argparse
import os

from lyric_whisper import LyricGen  # type: ignore # noqa
from prompt_generator import PromptGenerator, Music  # type: ignore # noqa
from tagging.tagger import Tagger  # type: ignore # noqa

argparser = argparse.ArgumentParser()
argparser.add_argument("--audio_dir", type=str, required=True)
argparser.add_argument("--output_dir", type=str, required=True)
args = argparser.parse_args()


class DSBuilder:
    """
    Class to build the dataset by generating prompts and lyrics for each audio file
    """

    def __init__(self, audio_dir: str, output_dir: str):
        self.audio_dir = audio_dir
        self.output_dir = output_dir

        self.lyric_gen = LyricGen()
        self.prompt_gen = PromptGenerator()

        self.tagger = Tagger(prob_threshold=0.2)
        self.tagger.load_model()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_data(self, prompt: str, file_path: str, lyric: str = ""):
        """Save the prompt and audio file to the output directory

        Args:
            prompt (str): the prompt to be saved
            file_path (str): the path to the audio file
            lyric (str, optional): the lyric of the audio. Defaults to "".
        """

        file_name = os.path.basename(file_path).split(".")[0]

        text = prompt + "\n\nLyric: " + lyric
        with open(
            os.path.join(self.output_dir, f"{file_name}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(text)

        os.system(f"cp {file_path} {os.path.join(self.output_dir, file_name)}.mp3")

    def build(self):
        """
        Build the dataset by generating prompts and lyrics for each audio file
        """

        batch = []
        for audio_path in os.listdir(self.audio_dir):
            audio_path = os.path.join(self.audio_dir, audio_path)

            batch.append(audio_path)

            if len(batch) < 2:
                continue

            out = self.tagger.tag(batch)

            batch = []

            for file_path, data in out.items():
                description = data[0]
                tags = data[1]

                name = os.path.basename(file_path).split(".")[0]
                music = Music(name, description, metadata=str(tags))

                prompt = self.prompt_gen.generate_prompt_from_music(music)

                prob, lyric = self.lyric_gen.generate_lyrics(file_path)

                if prob < 0.3:
                    lyric = ""

                self.save_data(prompt[0].content, file_path, lyric)


ds = DSBuilder(args.audio_dir, args.output_dir)
ds.build()
