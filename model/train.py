""" 
Train script for WaveAI
"""

import lightning as L
import torch
import torch.multiprocessing as mp
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loader import SynthDataset
from torch.utils.data import DataLoader, random_split

from model import WaveAILightning

lr_monitor = LearningRateMonitor(logging_interval="step")

torch.set_float32_matmul_precision("medium")
# torch.set_printoptions(threshold=10000)

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

model = WaveAILightning()

dataset = SynthDataset(audio_dir="/media/works/audio/", duration=30)

test_size = min(int(0.1 * len(dataset)), 200)
train_size = len(dataset) - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=dataset.collate_fn,
    num_workers=4,
    persistent_workers=True,
)
valid_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=dataset.collate_fn,
    num_workers=4,
    persistent_workers=True,
)

trainer = L.Trainer(
    limit_train_batches=100,
    max_epochs=100,
    callbacks=[lr_monitor, EarlyStopping(monitor="val_loss", mode="min")],
)

if __name__ == "__main__":
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
