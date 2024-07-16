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
        cross_att_hidden_size: int = 768,
        max_seq_length: int = 30_000,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        **kwargs,
    ):
        """Initialize the configuration class for the model

        Args:
            num_codebooks (int, optional): Number of codebooks to use in the model. Defaults to 9.
            codebook_size (int, optional): the number of vectors in each codebook. Defaults to 1024.. Defaults to 1024.
            hidden_size (int, optional): the dimension of the hidden_size to convert the index to a vector and process them. Defaults to 1024.
            cross_att_hidden_size (int, optional): the hidden size of the cross attention embedding. Defaults to 768 (T5).
            max_seq_length (int, optional): the maximum sequence length to generate. Defaults to 30_000.
            decoder_depth (int, optional): the number of decoder layers. Defaults to 4.
            decoder_heads (int, optional): the number of heads in the decoder. Defaults to 8.
            **kwargs: additional arguments
        """
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.cross_att_hidden_size = cross_att_hidden_size
        self.max_seq_length = max_seq_length
        self.decoder_depth = decoder_depth
        self.decoder_heads = decoder_heads

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
