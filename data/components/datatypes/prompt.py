from dataclasses import dataclass


@dataclass
class Prompt:
    """Class that represent the prompt that will be passed as an input for the music generation model.
    """

    content: str
