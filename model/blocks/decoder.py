""" 
Decoder block for the Transformer model
"""

import torch
import torch.nn as nn
from x_transformers import Decoder


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
        self,
        input_embds: torch.tensor,
        cross_att_embs: torch.tensor,
        attention_mask=None,
        padding_mask=None,
        **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            input_embds (torch.tensor): a tensor that represent the input embeddings of shape
                (batch_size, length, hidden_size)
            cross_att_embs (torch.tensor): a tensor that represent the cross attention embeddings
            attention_mask (torch.tensor, optional): a tensor that represent the attention mask. Defaults to None.
            padding_mask (torch.tensor, optional): a tensor that represent the padding mask. Defaults to None.
        Returns:
            torch.tensor: a tensor that represent the prob for each codebook idx
        """

        if cross_att_embs.size(-1) != self.config.hidden_size:
            cross_att_embs = self.cross_embd_proj(cross_att_embs)

        hidden_space = self.decoder(input_embds, context=cross_att_embs, **kwargs)
        lm_logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return lm_logits
