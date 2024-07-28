""" 
Decoder block
"""

import torch
import torch.nn as nn


class WaveAIDecoder(nn.Module):
    """Transformer decoder class for generate prediction of the next codebook idx"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_size, nhead=4, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

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
        input_embds: torch.Tensor,
        cross_att_embs: torch.Tensor,
        pattern_mask: torch.Tensor,
        **kwargs
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            input_embds (torch.tensor): a tensor that represent the input embeddings of shape
                (batch_size, length, hidden_size)
            cross_att_embs (torch.tensor): a tensor that represent the cross attention embeddings
            pattern_mask (torch.tensor): a tensor that represent the pattern mask
        Returns:
            torch.tensor: a tensor that represent the prob for each codebook idx
        """

        if cross_att_embs.size(-1) != self.config.hidden_size:
            cross_att_embs = self.cross_embd_proj(cross_att_embs)

        # use .generate_square_subsequent_mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            input_embds.size(1)
        ).to(input_embds.device)

        padding_mask = (pattern_mask == self.config.codebook_size).all(dim=1)[
            :, : input_embds.size(1)
        ]

        hidden_space = self.transformer_decoder(
            tgt=input_embds,
            memory=cross_att_embs,
            tgt_mask=causal_mask,
        )

        lm_logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return lm_logits
