from dataclasses import dataclass


@dataclass
class Lyrics:
    """Class that represent all data about the lyrics that will be used as the
    input of the prompt generator class.
    """

    content: str
