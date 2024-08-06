""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn

from .blocks.decoder import WaveAIDecoder
from .layers.positional_encoding import PositionalEncoding


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(self.config.codebook_size + 1, self.config.hidden_size)
                for _ in range(self.config.num_codebooks)
            ]
        )  # each codebook has his own embedding layer

        self.decoder = WaveAIDecoder(self.config)
        self.position_embedding = PositionalEncoding(
            self.config.hidden_size, self.config.max_seq_length
        )

    def prepare_inputs_for_generation(self, batch_size: int = 1) -> torch.Tensor:
        """Create the initial input for the decoder

        Args:
            batch_size (int, optional): the batch size. Defaults to 1.

        Returns:
            torch.Tensor: a tensor that represent the initial input for the decoder
        """

        decoder_input_ids_start = (
            torch.ones((batch_size, self.config.num_codebooks, 1), dtype=torch.long)
            * self.config.pad_token_id
        )

        return decoder_input_ids_start

    @staticmethod
    def shift_tokens_right(
        input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ):
        """
        Shift input ids one token to the right.
        """
        # transpose to get (bsz, num_codebooks, seq_len)
        # input_ids = input_ids.transpose(1, 2)
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError(
                "Make sure to set the decoder_start_token_id attribute of the model's configuration."
            )
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError(
                "Make sure to set the pad_token_id attribute of the model's configuration."
            )
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
        self, input_ids: torch.Tensor, cross_att_emb: torch.Tensor = None, **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            input_ids (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            cross_att_emb (torch.tensor | None): a tensor that represent the cross attention embedding of the prompt
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        # embed the codebook idx
        inputs_embed = [
            self.embedding_layers[codebook_idx](input_ids[:, codebook_idx])
            for codebook_idx in range(self.config.num_codebooks)
        ]

        # sum the embeddings of each codebook idx
        inputs_embed = sum(inputs_embed)  # dim: (batch_size, length, hidden_size)

        inputs_embed = self.position_embedding(inputs_embed)

        # pass the embeddings through the decoder
        logits = self.decoder(inputs_embed, cross_att_emb)

        return logits
