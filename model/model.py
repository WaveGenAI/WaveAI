""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn
from x_transformers import Decoder, MultiInputTransformerWrapper


class PositionalEncoding(nn.Module):
    """Layer that add the sinusoidal positionnal encoding"""

    def __init__(self, dim_model: int, max_seq_len: int):
        """Initialize the PositionalEncoding layer

        Args:
            dim_model (int): the model dimension
            max_seq_len (int): the maximum sequence length
        """

        super().__init__()

        pe = torch.zeros((1, max_seq_len, dim_model))

        pos = torch.arange(max_seq_len).unsqueeze(1)
        divid = 10_000 ** (torch.arange(0, dim_model, 2) / dim_model)

        pe[0, :, 0::2] = torch.sin(pos / divid)
        pe[0, :, 1::2] = torch.cos(pos / divid)

        self._pe = pe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """A method that adds the positional encoding to the input tensor

        Args:
            inputs (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the input tensor with the positional encoding
        """

        self._pe = self._pe.to(inputs.device)
        return inputs + self._pe[:, : inputs.shape[1], :]


class WaveAI(nn.Module):
    """WaveAI model class"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.prepends_embedding = nn.Embedding(
            self.config.model.vocab_size, self.config.model.hidden_size
        )

        if self.config.model.cross_att_hidden_size != self.config.model.hidden_size:
            self.cross_embd_proj = nn.Linear(
                self.config.model.cross_att_hidden_size, self.config.model.hidden_size
            )  # if text encoder returns a different hidden size than the model hidden size

        embds = {
            k: self.config.model.codebook_size + 1
            for k in range(self.config.model.num_codebooks)
        }

        self.decoder = MultiInputTransformerWrapper(
            num_tokens=embds,
            max_seq_len=self.config.model.max_seq_length,
            return_only_embed=True,
            attn_layers=Decoder(
                dim=self.config.model.hidden_size,
                depth=self.config.model.decoder_depth,
                heads=self.config.model.decoder_heads,
                attn_flash=True,
                rotary_pos_emb=True,
                layer_dropout=0.1,  # stochastic depth - dropout entire layer
                attn_dropout=0.1,  # dropout post-attention
                ff_dropout=0.1,  # feedforward dropout
                use_scalenorm=True,
                cross_attend=True,
            ),
        )

        # each head predicts a codebook (not its index)
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.config.model.hidden_size, self.config.model.codebook_size
                )
                for _ in range(self.config.model.num_codebooks)
            ]
        )

        self.pos_enc = PositionalEncoding(
            self.config.model.hidden_size, self.config.model.max_lyrics_length
        )

    def forward(
        self,
        inputs_ids: torch.Tensor,
        cross_att_embs: torch.Tensor,
        prepends_ids: torch.Tensor,
    ) -> torch.tensor:
        """Forward pass through the model

        Args:
            inputs_ids (torch.tensor): a tensor that represent the codebook idx of shape
                (batch_size, num_codebooks, length)
            cross_att_embs (torch.tensor): a tensor that represent the cross attention embedding of the prompt
            prepends_ids (torch.tensor): a tensor that represent the prepends idx of shape
        Returns:
            torch.tensor: a tensor that represent the logits prob
        """

        inputs_ids = inputs_ids.masked_fill(
            inputs_ids == -100, self.config.model.pad_token_id
        )

        # create an embedding for each codebook
        x = {}
        for k in range(self.config.model.num_codebooks):
            x[k] = inputs_ids[:, k, :]

        # get the prepends embeddings
        prepends_embds = self.prepends_embedding(prepends_ids)

        # add the positional encoding to the prepends embeddings
        prepends_embds = self.pos_enc(prepends_embds)

        # project the cross attention embeddings if needed
        if cross_att_embs.size(-1) != self.config.model.hidden_size:
            cross_att_embs = self.cross_embd_proj(cross_att_embs)

        # pass the embeddings through the decoder
        hidden_space = self.decoder(
            x, prepend_embeds=prepends_embds, context=cross_att_embs
        )

        logits = torch.stack(
            [lm_head(hidden_space) for lm_head in self.lm_heads], dim=1
        )

        return logits
