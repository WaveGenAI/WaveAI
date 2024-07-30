""" 
Train script for WaveAI
"""

import lightning as L
import torch
from config import Config
from lightning.pytorch.callbacks import LearningRateMonitor
from loader import SynthDataset
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from model import WaveAI

lr_monitor = LearningRateMonitor(logging_interval="step")
# torch.set_printoptions(threshold=10000)


class WaveAILightning(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.config = Config()
        self.model = WaveAI(self.config)

    def training_step(self, batch, batch_idx):
        tgt_audio = batch[0]
        src_text = batch[1]

        tgt_audio = tgt_audio[
            :, :, : self.config.max_seq_length - self.config.num_codebooks + 1
        ]  # cut the audio to the max length (including the codebooks because of the delay pattern)

        logits, labels = self.model(tgt_audio, src_text)

        loss = torch.zeros([])

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss_fn = CrossEntropyLoss()
        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            # get index of the most probable token
            max_prob_idx = logits_k.argmax(dim=-1)
            # print(targets_k, max_prob_idx)

            loss += loss_fn(logits_k.cpu(), targets_k.cpu())

        loss = loss / self.config.num_codebooks

        print(f"Loss: {loss}")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tgt_audio = batch[0]
        src_text = batch[1]

        tgt_audio = tgt_audio[
            :, :, : self.config.max_seq_length - self.config.num_codebooks + 1
        ]  # cut the audio to the max length (including the codebooks because of the delay pattern)

        logits, labels = self.model(tgt_audio, src_text)

        loss = torch.zeros([])

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss_fn = CrossEntropyLoss()
        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            loss += loss_fn(logits_k.cpu(), targets_k.cpu())

        loss = loss / self.config.num_codebooks

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=1e-6, total_iters=5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


if __name__ == "__main__":
    model = WaveAILightning()

    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn
    )

    trainer = L.Trainer(limit_train_batches=100, max_epochs=100, callbacks=[lr_monitor])
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
