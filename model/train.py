""" 
Train script for WaveAI
"""

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

        labels = labels[:, :, : self.config.max_seq_length]

        logits = self.model(None, src_text)

        loss = torch.zeros([])

        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, card]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            loss += self._loss_fn(logits_k.cpu(), targets_k.cpu())

        loss = loss / self.config.num_codebooks

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = WaveAILightning()

    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn
    )

    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=100,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
