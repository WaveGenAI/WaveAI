import torch
import torch.nn as nn


class WaveAIDecoder(nn.Module):
    """Transformer decoder class for generating prediction of the next codebook idx"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.decoder_heads,
            dim_feedforward=4 * self.config.hidden_size,  # typically 4x the hidden size
            dropout=0.1,  # You might want to make this configurable
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.config.decoder_depth
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
        padding_mask: torch.Tensor,
        cross_att_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model
        Args:
            input_embds (torch.tensor): a tensor representing the input embeddings of shape
                (batch_size, length, hidden_size)
            padding_mask (torch.tensor): a tensor representing the padding mask of the input embeddings
            cross_att_embs (torch.tensor | None): a tensor representing the cross-attention embedding of the prompt
        Returns:
            torch.tensor: a tensor representing the prob for each codebook idx
        """
        if cross_att_embs is None:
            cross_att_embs = torch.zeros(
                input_embds.size(0),
                input_embds.size(1),
                self.config.cross_att_hidden_size,
            ).to(input_embds.device)

        if cross_att_embs.size(-1) != self.config.hidden_size:
            cross_att_embs = self.cross_embd_proj(
                cross_att_embs.float()
            )  # project the cross-attention embedding to the model hidden size

        # Convert padding_mask to attention mask
        attention_mask = ~padding_mask.bool()

        # PyTorch's TransformerDecoder expects the target sequence mask to be of shape (sequence_length, sequence_length)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            input_embds.size(1)
        ).to(input_embds.device)

        # Pass the input embeddings through the PyTorch TransformerDecoder with the cross-attention embeddings
        hidden_space = self.transformer_decoder(
            tgt=input_embds,
            memory=cross_att_embs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=attention_mask,
        )

        # each head predicts a codebook
        lm_logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return lm_logits
