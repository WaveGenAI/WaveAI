""" 
Decoder block
"""

import torch
import torch.nn as nn
from x_transformers import Decoder


class WaveAIDecoder(nn.Module):
    """Transformer decoder class for generating prediction of the next codebook idx"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer_decoder = Decoder(
            dim=self.config.hidden_size,
            depth=self.config.decoder_depth,
            heads=self.config.decoder_heads,
            cross_attend=self.config.cross_att,  # cross-attention state
            attn_flash=True,
        )

        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, self.config.codebook_size)
                for _ in range(self.config.num_codebooks)
            ]
        )  # each head predicts a codebook (not its index)

        self.cross_embd_proj = nn.Linear(
            self.config.cross_att_hidden_size, self.config.hidden_size
        )  # if text encoder returns a different hidden size than the model hidden size

    def forward(
        self,
        input_embds: torch.Tensor,
        cross_att_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model

        Args:
            input_embds (torch.tensor): a tensor representing the input embeddings of shape
                (batch_size, length, hidden_size)
            cross_att_embs (torch.tensor | None): a tensor representing the cross-attention embedding of the prompt
        Returns:
            torch.tensor: a tensor representing the prob for each codebook idx
        """

        if (
            cross_att_embs is not None
            and cross_att_embs.size(-1) != self.config.hidden_size
        ):
            cross_att_embs = self.cross_embd_proj(
                cross_att_embs
            )  # project the cross-attention embedding to the model hidden size

        # Pass the input embeddings through the x-transformer decoder with the cross-attention embeddings
        if cross_att_embs is not None:
            hidden_space = self.transformer_decoder(
                x=input_embds, context=cross_att_embs
            )
        else:
            hidden_space = self.transformer_decoder(x=input_embds)

        # each head predicts a codebook
        lm_logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return lm_logits
