""" 
Encoder Model for Text Data
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer


class T5EncoderBaseModel(nn.Module):
    """
    T5 Encoder Model, which is used to encode the input text into a latent space.
    """

    def __init__(self, name: str = "google-t5/t5-small", max_length: int = 512):
        """Encoder Model for Text Data

        Args:
            name (str, optional): the name of the model to use. Defaults to "t5-base".
            max_length (int, optional): the max length of the prompt. Defaults to 512.
        """
        super().__init__()

        self.encoder = T5EncoderModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self._max_length = max_length

    def forward(self, inputs: list) -> torch.Tensor:
        """Forward pass of the model

        Args:
            inputs (list): the input text

        Returns:
            torch.Tensor: the latent space of the input text
        """

        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).input_ids
        input_ids = input_ids.to(self.encoder.device)

        latent_space = self.encoder(input_ids).last_hidden_state

        return latent_space
