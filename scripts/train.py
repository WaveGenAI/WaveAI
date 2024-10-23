import argparse

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from waveai.audio_processor import AudioProcessor
from waveai.trainer import Trainer
from waveai.utils.config_parser import ConfigParser
from waveai.utils.logs import LogPredictionSamplesCallback

args = argparse.ArgumentParser()
args.add_argument(
    "--config_path", type=str, required=True, help="Path to the configuration file"
)
args.add_argument(
    "--save_path",
    type=str,
    default="./checkpoints",
    required=False,
    help="Path to save the model",
)
args.add_argument(
    "--load_from",
    type=str,
    default=None,
    required=False,
    help="Path to load the model (for finetuning)",
)
args = args.parse_args()

# opti
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = ConfigParser(config_path=args.config_path)
lr_monitor = LearningRateMonitor(logging_interval="step")
audio_processor = AudioProcessor(config)
model = Trainer(config, audio_processor)

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.normal_(p, 0, 0.02)


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset(config.data.dataset_id, split="train")
    dataset = dataset.train_test_split(test_size=min(len(dataset) * 0.1, 2000))

    if config.train.shuffle_data:
        dataset["train"] = dataset["train"].shuffle(seed=42)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=config.train.batch_size,
        num_workers=config.train.train_num_workers,
        collate_fn=audio_processor.collate_fn,
        pin_memory=True,
        shuffle=config.train.shuffle_data,
    )

    valid_dataloader = DataLoader(
        dataset["test"],
        batch_size=config.train.batch_size,
        num_workers=config.train.val_num_workers,
        collate_fn=audio_processor.collate_fn,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(project="WAVEAI", log_model=True)
    wandb_logger.watch(model)

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        callbacks=[lr_monitor, LogPredictionSamplesCallback()],
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        gradient_clip_val=config.train.gradient_clip_val,
        logger=wandb_logger,
        log_every_n_steps=1,
        default_root_dir=args.save_path,
        precision="bf16-mixed",
        profiler="simple",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=args.load_from,
    )
