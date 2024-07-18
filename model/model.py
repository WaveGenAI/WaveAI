""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn

from model.blocks.decoder import WaveAIDecoder
from model.layers.positional_encoding import PositionalEncoding


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
        )

        self.decoder = WaveAIDecoder(self.config)
        self.position_embedding = PositionalEncoding(
            self.config.hidden_size, self.config.max_seq_length
        )

    def build_delay_pattern(
        self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int = None
    ) -> tuple:
        """Build the delay pattern for the input ids

        Args:
            input_ids (torch.LongTensor): a tensor that represent the codebook idx of shape
            pad_token_id (int): the padding token id
            max_length (int, optional): the maximum length of the input. Defaults to None.

        Returns:
            tuple: a tuple of (input_ids, padding_mask)
        """

        bsz, num_codebooks, seq_len = input_ids.shape

        input_ids_shifted = (
            torch.ones(
                (bsz, num_codebooks, max_length),
                dtype=torch.long,
                device=input_ids.device,
            )
            * pad_token_id
        )

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(num_codebooks):
            input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[
                :, codebook
            ]

        # create the padding mask
        padding_mask = input_ids_shifted == pad_token_id

        return input_ids, padding_mask

    def create_attention_mask(self, input_ids: torch.tensor) -> torch.tensor:
        """Create the attention mask for the input ids

        Args:
            input_ids (torch.tensor): a tensor that represent the codebook idx of shape

        Returns:
            torch.tensor: a tensor that represent the attention mask
        """
        bsz, num_codebooks, seq_len = input_ids.shape
        attention_mask = (
            torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        attention_mask = attention_mask.expand(bsz, num_codebooks, -1, -1)
        return attention_mask.bool()

    def forward(
        self, input_ids: torch.tensor, cross_att_emb: torch.tensor, **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            input_ids (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            cross_att_emb (torch.tensor): a tensor that represent the cross attention embedding of the prompt
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        input_ids, padding_mask = self.build_delay_pattern(
            input_ids,
            pad_token_id=self.config.codebook_size,
            max_length=self.config.max_seq_length,
        )

        attention_mask = self.create_attention_mask(input_ids)

        # embedding
        inputs_embed = [
            self.embedding_layers[codebook_idx](input_ids[:, codebook_idx])
            for codebook_idx in range(self.config.num_codebooks)
        ]

        inputs_embed = sum(inputs_embed)  # dim: (batch_size, length, hidden_size)
        inputs_embed = self.position_embedding(inputs_embed)

        logits = self.decoder(
            inputs_embed,
            cross_att_emb,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )

        return logits
