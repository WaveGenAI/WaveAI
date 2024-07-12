""" 
Class for the main model (music generation)
"""

import torch
import torch.nn as nn


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

    def forward(self, input_ids, **kwargs):
        raise NotImplementedError
