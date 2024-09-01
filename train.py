import argparse

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import text_encoder
from model.config import Config
from model.lightning_model import WaveAILightning

args = argparse.ArgumentParser()
args.add_argument("--save_path", type=str, default="checkpoints", required=False)
args.add_argument("--checkpoint_path", type=str, default=None, required=False)
args = args.parse_args()


torch.set_float32_matmul_precision("medium")
config = Config()

lr_monitor = LearningRateMonitor(logging_interval="step")
tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
text_enc = text_encoder.T5EncoderBaseModel().eval()
model = WaveAILightning()

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.normal_(p, 0, 0.02)


def stereo_convert(codec: torch.Tensor) -> torch.Tensor:
    """Convert the codec tensor to stereo if needed

    Args:
        codec (torch.Tensor): the codec tensor

    Returns:
        torch.Tensor: the stereo codec tensor if needed
    """

    # if the shape is 2D, add a channel dimension
    if len(codec.shape) == 2:
        codec = codec.unsqueeze(-1)

    # if the codec is mono and the model is stereo, duplicate the channel
    if codec.shape[-1] == 1 and config.model.stereo:
        codec = torch.cat([codec, codec], dim=-1)

    return codec


# Define the collate function for processing batches
@torch.no_grad()
def collate_fn(rows):
    codec = [stereo_convert(torch.permute(row["codec"], (2, 1, 0))) for row in rows]
    prompt = [row["prompt"] for row in rows]
    lyrics = [row["lyrics"] for row in rows]

    # Pad the codec tensors and stack them
    codes = torch.nn.utils.rnn.pad_sequence(codec, batch_first=True, padding_value=-100)

    codes = codes.permute(0, 3, 2, 1).contiguous()

    # convert from batch x channel x num_codebooks x seq_length to batch x (num_codebooks x channels) x seq_length
    codes = codes.view(codes.size(0), -1, codes.size(-1))

    # convert codes to long
    codes = codes.long()

    # tokenize the lyrics
    lyrics_ids = tokenizer(
        lyrics,
        padding=True,
        truncation=True,
        max_length=config.data.max_lyrics_length,
        return_tensors="pt",
    ).input_ids

    # encode the prompt
    prompts_embeds, prompts_masks = text_enc(prompt)

    return codes, prompts_embeds, prompts_masks, lyrics_ids


if __name__ == "__main__":
    # Load the dataset
    train_ds, test_ds = load_dataset(config.data.dataset_id, split=["train", "test"])
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")

    # shuffle the dataset
    train_ds = train_ds.shuffle(seed=42)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        num_workers=config.train.train_num_workers,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        test_ds,
        batch_size=config.train.batch_size,
        num_workers=config.train.val_num_workers,
        collate_fn=collate_fn,
    )

    wandb_logger = WandbLogger(project="WAVEAI")

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        callbacks=[lr_monitor],
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        gradient_clip_val=config.train.gradient_clip_val,
        logger=wandb_logger,
        log_every_n_steps=1,
        default_root_dir=args.save_path,
        precision="16-mixed",
        profiler="simple",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=args.checkpoint_path,
    )
