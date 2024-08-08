""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn
from x_transformers import Decoder, MultiInputTransformerWrapper


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

        embds = {
            k: self.config.codebook_size + 1 for k in range(self.config.num_codebooks)
        }

        self.decoder = MultiInputTransformerWrapper(
            num_tokens=embds,
            max_seq_len=self.config.max_seq_length,
            return_only_embed=True,
            attn_layers=Decoder(
                dim=self.config.hidden_size,
                depth=self.config.decoder_depth,
                heads=self.config.decoder_heads,
                cross_attend=self.config.cross_att,  # cross-attention state
                attn_flash=True,
                rotary_pos_emb=True,
            ),
        )

        # each head predicts a codebook (not its index)
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, self.config.codebook_size)
                for _ in range(self.config.num_codebooks)
            ]
        )

    def prepare_inputs_for_generation(self, batch_size: int = 1) -> torch.Tensor:
        """Create the initial input for the decoder

        Args:
            batch_size (int, optional): the batch size. Defaults to 1.

        Returns:
            torch.Tensor: a tensor that represent the initial input for the decoder
        """

        decoder_inputs_ids_ids_start = (
            torch.ones((batch_size, self.config.num_codebooks, 1), dtype=torch.long)
            * self.config.pad_token_id
        )

        return decoder_inputs_ids_ids_start

    def forward(
        self, inputs_ids: torch.Tensor, cross_att_emb: torch.Tensor = None, **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            inputs_ids (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            cross_att_emb (torch.tensor | None): a tensor that represent the cross attention embedding of the prompt
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        inputs_ids = inputs_ids.masked_fill(
            inputs_ids == -100, self.config.pad_token_id
        )

        x = {}
        for k in range(self.config.num_codebooks):
            x[k] = inputs_ids[:, k, :]

        # pass the embeddings through the decoder
        hidden_space = self.decoder(x)

        logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return logits
