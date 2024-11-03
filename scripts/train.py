import argparse

import lightning as L
import torch
from datasets import Dataset, load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from waveai.audio_processor import AudioProcessor
from waveai.trainer import Trainer
from waveai.utils.config_parser import ConfigParser
from waveai.utils.logs import LogPredictionSamplesCallback
from waveai.utils.utils import load_webdataset

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

config = ConfigParser(config_path=args.config_path)
lr_monitor = LearningRateMonitor(logging_interval="step")
audio_processor = AudioProcessor(config)
model = Trainer(config, audio_processor)

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.normal_(p, 0, 0.02)


# skip samples that are too short
def select(sample):
    audio = sample[config.data.audio_column]

    if audio.shape[-1] < (config.model.num_codebooks * 2):
        return None

    return sample


if __name__ == "__main__":
    # Load the dataset
    dataset_train = load_webdataset(
        config.data.dataset_id,
        "train",
        map=select,
        shuffle=config.data.shuffle_data and not config.train.debug,
    ).with_length(config.data.train_size)

    dataset_test = load_webdataset(
        config.data.dataset_id, "test", map=select, shuffle=False
    )

    # args for the trainer
    kwargs = {}

    if config.train.debug:
        first_row = next(iter(dataset_train))
        duplicated_data = Dataset.from_dict(
            {key: [value] * 500 for key, value in first_row.items()}
        )
        dataset_train = duplicated_data

        # disable validation
        kwargs["limit_val_batches"] = 0
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=config.train.batch_size,
        num_workers=config.train.train_num_workers,
        collate_fn=audio_processor.collate_fn,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        dataset_test,
        batch_size=config.train.batch_size,
        num_workers=config.train.val_num_workers,
        collate_fn=audio_processor.collate_fn,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(
        project="WAVEAI", log_model=True, save_dir=args.save_path
    )
    wandb_logger.watch(model)

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        callbacks=[lr_monitor, LogPredictionSamplesCallback()],
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        gradient_clip_val=config.train.gradient_clip_val,
        logger=wandb_logger,
        log_every_n_steps=1,
        default_root_dir=args.save_path,
        precision=16,
        profiler="simple",
        **kwargs,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=args.load_from,
    )
