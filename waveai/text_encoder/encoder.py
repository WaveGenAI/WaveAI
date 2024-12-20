""" 
Encoder Model for Text Data
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class T5EncoderBaseModel(nn.Module):
    """
    T5 Encoder Model, which is used to encode the input text into a latent space.
    """

    def __init__(self, name: str = "google-t5/t5-large", max_seq_len: int = 1024):
        """Encoder Model for Text Data

        Args:
            name (str, optional): the name of the model to use. Defaults to "t5-base".
        """
        super().__init__()

        self.encoder = T5EncoderModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self._max_seq_len = max_seq_len

    @torch.no_grad()
    def forward(self, inputs: list) -> torch.Tensor:
        """Forward pass of the model

        Args:
            inputs (list): the input text

        Returns:
            torch.Tensor: the latent space of the input text
        """

        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_seq_len,
        )

        input_ids = encoded["input_ids"]
        padding_mask = encoded["attention_mask"]

        embeddings = self.encoder(input_ids=input_ids, attention_mask=padding_mask)[
            "last_hidden_state"
        ]

        return embeddings, padding_mask.bool()
