""" 
Train script for WaveAI
"""

from typing import Any
import lightning as L
from loader import SynthDataset
from torch.utils.data import DataLoader


class WaveAI(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


if __name__ == "__main__":
    # wave_ai = WaveAI()
    dataset = SynthDataset(audio_dir="/media/works/waveai_music/")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        print(batch)
        break
