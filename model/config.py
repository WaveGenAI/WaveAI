"""
Configuration file for the model
"""


class Config:
    """
    Configuration class for the model
    """

    def __init__(
        self,
        num_codebooks: int = 9,
        codebook_size: int = 1024,
        hidden_size: int = 1024,
        **kwargs,
    ):
        """Initialize the configuration class for the model

        Args:
            num_codebooks (int, optional): Number of codebooks to use in the model. Defaults to 9.
            codebook_size (int, optional): the number of vectors in each codebook. Defaults to 1024.. Defaults to 1024.
            hidden_size (int, optional): the dimension of the hidden_size to convert the index to a vector and process them. Defaults to 1024.
            **kwargs: additional arguments
        """
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
