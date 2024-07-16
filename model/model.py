""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder


class WaveAIDecoder(nn.Module):
    """Transformer decoder class for generate prediction of the next codebook idx"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(
            dim=config.hidden_size,
            depth=config.decoder_depth,
            heads=config.decoder_heads,
            cross_attend=True,
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, self.config.codebook_size)
                for _ in range(self.config.num_codebooks)
            ]
        )  # each head predict a codebook (not his index)

        self.cross_embd_proj = nn.Linear(
            self.config.cross_att_hidden_size, self.config.hidden_size
        )  # if text encoder return different hidden size than the model hidden size

    def forward(
        self, input_embds: torch.tensor, cross_att_embs: torch.tensor, **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            input_embds (torch.tensor): a tensor that represent the input embeddings of shape
                (batch_size, length, hidden_size)
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        if cross_att_embs.size(-1) != self.config.hidden_size:
            cross_att_embs = self.cross_embd_proj(cross_att_embs)

        hidden_space = self.decoder(input_embds, context=cross_att_embs)
        lm_logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return lm_logits


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(self.config.codebook_size, self.config.hidden_size)
                for _ in range(self.config.num_codebooks)
            ]
        )

        self.decoder = WaveAIDecoder(self.config)

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

        # Embedding
        inputs_embed = [
            embedding_layer(input_ids[:, codebook_idx])
            for codebook_idx, embedding_layer in enumerate(self.embedding_layers)
        ]

        # TODO: Intervaling pattern

        inputs_embed = sum(inputs_embed)  # dim: (batch_size, length, hidden_size)

        logits = self.decoder(inputs_embed, cross_att_emb)

        return logits
