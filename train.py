""" 
Train script for WaveAI
"""

import lightning as L
import torch
import torch.multiprocessing as mp
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from model.config import Config
from model.lightning_model import WaveAILightning
from model.loader import SynthDataset

lr_monitor = LearningRateMonitor(logging_interval="step")
config = Config()

torch.set_float32_matmul_precision("medium")
# torch.set_printoptions(threshold=10000)

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

model = WaveAILightning()

dataset = SynthDataset(overwrite=False)

test_size = min(int(0.1 * len(dataset)), 200)
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.collate_fn,
    num_workers=0,
)
valid_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=dataset.collate_fn,
    num_workers=0,
)

wandb_logger = WandbLogger(project="WAVEAI")

trainer = L.Trainer(
    max_epochs=config.train.max_epochs,
    callbacks=[lr_monitor, EarlyStopping(monitor="val_loss", mode="min")],
    accumulate_grad_batches=config.train.accumulate_grad_batches,
    gradient_clip_val=config.train.gradient_clip_val,
    logger=wandb_logger,
    log_every_n_steps=1,
    default_root_dir="checkpoints",
    precision="16-mixed",
)


if __name__ == "__main__":
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
