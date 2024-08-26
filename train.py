import lightning as L
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import text_encoder
from model.config import Config
from model.lightning_model import WaveAILightning

torch.set_float32_matmul_precision("medium")
config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_monitor = LearningRateMonitor(logging_interval="step")
tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
text_enc = (
    text_encoder.T5EncoderBaseModel(max_length=config.data.max_prompt_length)
    .eval()
    .to(device)
)
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
    prompts_embeds = text_enc(prompt)

    return codes, prompts_embeds, lyrics_ids


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Load the dataset
    dataset = load_dataset(config.data.dataset_id)
    dataset = dataset.with_format("torch")

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=config.train.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        dataset["test"],
        batch_size=config.train.batch_size,
        num_workers=2,
        collate_fn=collate_fn,
        persistent_workers=True,
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
        devices="auto",
        strategy="auto",
        profiler="simple",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
