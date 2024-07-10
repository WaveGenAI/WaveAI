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

    def __init__(self, name: str = "t5-base"):
        super().__init__()

        self.encoder = T5EncoderModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def forward(self, inputs: str) -> torch.Tensor:
        """Forward pass of the model

        Args:
            inputs (str): the input text

        Returns:
            torch.Tensor: the latent space of the input text
        """

        input_ids = self.tokenizer(inputs, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.encoder.device)

        latent_space = self.encoder(input_ids)

        return latent_space
