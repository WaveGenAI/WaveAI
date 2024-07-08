from dataclasses import dataclass

from lyrics import Lyrics


@dataclass
class Music:
    """Class that represent all data about a music that will be used as the
    input of the prompt generator class.
    """

    name: str
    clap_desc: str
    metadata: str = "No metadata available"
    instruction_id: int = 1
    lyrics: Lyrics = None
    transformed_lyrics: Lyrics = None
