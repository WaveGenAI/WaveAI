""" 
Train script for WaveAI
"""

from typing import Any

import lightning as L
import torch
from config import Config
from loader import SynthDataset
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from model import WaveAI


class WaveAILightning(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.config = Config()
        self._loss_fn = CrossEntropyLoss()
        self.model = WaveAI(self.config)

    def training_step(self, batch, batch_idx):
        labels = batch[0]
        src_text = batch[1]

        bsz, num_codebooks, seq_len = labels.shape

        labels = labels[:, :, : self.config.max_seq_length]

        logits = self.model(None, src_text)
        print(logits.shape)
        loss = torch.zeros([])

        # per codebook cross-entropy
        # ref: https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/solvers/musicgen.py#L242-L243
        for codebook in range(self.config.num_codebooks):
            codebook_logits = (logits[:, codebook].view(-1, logits.shape[-1])).cpu()
            codebook_labels = labels[:, codebook].view(-1).cpu()

            loss += self._loss_fn(codebook_logits, codebook_labels)

        loss = loss / self.config.num_codebooks
        print(f"Loss: {loss}")
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = WaveAILightning()

    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")
    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn
    )

    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
