import tempfile

import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics import Metric

import wandb


class LogPredictionSamplesCallback(Callback):
    """Log the predicted samples"""

    def __init__(self) -> None:
        super().__init__()
        self.sample_table = wandb.Table(columns=["prompt", "audio"])
        self.temp_dir = tempfile.TemporaryDirectory()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""

        if "predictions" in outputs and outputs["predictions"] is not None:
            outputs["predictions"].write(f"{self.temp_dir.name}/test_{batch_idx}.wav")

            self.sample_table.add_data(
                batch[4][0],  # batch[4] is the prompt, [0] to get the first element
                wandb.Audio(f"{self.temp_dir.name}/test_{batch_idx}.wav"),
            )

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation ends."""

        wandb.log({f"eval-epoch-{trainer.current_epoch}": self.sample_table})
        self.sample_table = wandb.Table(columns=["prompt", "audio"])

        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "predictions" in outputs and outputs["predictions"] is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                outputs["predictions"].write(f.name)
                wandb.log(
                    {
                        "audio": wandb.Audio(f.name, caption=batch[3][0]),
                    }
                )


class LossTensor(Metric):
    """
    Loss with accumulation of gradients.
    """

    def __init__(self):
        super().__init__()
        self.add_state("loss", torch.tensor(0, dtype=torch.float))
        self.add_state("counter", torch.tensor(0, dtype=torch.float))

    def update(self, loss):
        self.loss += loss.clone().detach().cpu()
        self.counter += 1

    def compute(self):
        return self.loss / self.counter
