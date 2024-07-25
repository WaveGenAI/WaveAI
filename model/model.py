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
        padding_mask = (input_ids_shifted == pad_token_id).all(dim=1)

        # pad input_ids to max_length
        input_ids = input_ids_shifted

        return input_ids, padding_mask

    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create an attention mask for the decoder

        Args:
            seq_len (int): the length of the sequence
            device (torch.device): the device to place the mask on

        Returns:
            torch.Tensor: the attention mask
        """
        attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).view(
            seq_len, seq_len
        )

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

        # embedding
        inputs_embed = [
            self.embedding_layers[codebook_idx](input_ids[:, codebook_idx])
            for codebook_idx in range(self.config.num_codebooks)
        ]

        inputs_embed = sum(inputs_embed)  # dim: (batch_size, length, hidden_size)
        inputs_embed = self.position_embedding(inputs_embed)

        attention_mask = self.create_attention_mask(
            seq_len=inputs_embed.size(1), device=input_ids.device
        )

        logits = self.decoder(
            inputs_embed,
            cross_att_emb,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
        )

        return logits
