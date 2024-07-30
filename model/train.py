""" 
Train script for WaveAI
"""

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loader import SynthDataset
from torch.utils.data import DataLoader, random_split

from model import WaveAILightning

lr_monitor = LearningRateMonitor(logging_interval="step")
# torch.set_printoptions(threshold=10000)

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

trainer = L.Trainer(
    limit_train_batches=100,
    max_epochs=100,
    callbacks=[lr_monitor, EarlyStopping(monitor="val_loss", mode="min")],
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
