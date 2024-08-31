import tempfile

import bitsandbytes as bnb
import lightning as L
import torch
from audiotools import AudioSignal
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torchmetrics import Metric

import wandb
from model.audio_autoencoder import DAC as audio_autoencoder

from .config import Config
from .generation import Generation
from .model import WaveAI
from .pattern import DelayPattern

config = Config()


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


class WaveAILightning(L.LightningModule):
    def __init__(self) -> None:
        """Lightning module for WaveAI.

        Args:
            use_prompt (bool, optional): is model trained on prompt. Defaults to False.
        """

        super().__init__()

        self.config = config
        self.delay_pattern = DelayPattern()

        self.num_codebooks = self.config.model.num_codebooks
        if self.config.model.stereo:
            self.num_codebooks = self.num_codebooks * 2

        self.model = WaveAI(
            self.num_codebooks,
            self.config.model.codebook_size,
            self.config.model.hidden_size,
            self.config.model.decoder_depth,
            self.config.model.decoder_heads,
            self.config.model.memory_dim,
        )
        self.save_hyperparameters()

        # put in list to avoid training in the pytorch-lightning (a bit hacky)
        self.loss_metric = [LossTensor()]
        self.loss_fn = [CrossEntropyLoss()]

        self.generator = Generation(
            self.model,
            self.num_codebooks,
            self.config.model.pad_token_id,
            self.config.model.stereo,
        )
        self.audio_codec = audio_autoencoder()
        self.audio_codec.model.to("cpu")

    def step(self, batch, batch_idx) -> torch.Tensor:
        audio, prompts, prompts_masks, _ = batch

        # cut the audio to the max length
        audio = audio[:, :, : self.config.model.max_seq_length]

        # just for logging (to see the number of tokens)
        self.log("nbm_token", audio.numel())

        # get the delay pattern, in this way each token is delayed by the same amount of time
        tokens, _ = self.delay_pattern.build_delay_pattern_mask(
            audio, self.config.model.pad_token_id, self.config.model.max_seq_length
        )

        # shift the tokens to the right (like that the model will predict the next token and will not see the future)
        tokens = self.delay_pattern.shift_tokens_right(
            tokens, self.config.model.pad_token_id, self.config.model.pad_token_id
        )

        # create the inputs and labels tensors
        inputs_ids = tokens[..., :-1]
        labels = tokens[..., 1:]

        logits = self.model(inputs_ids, prompts, prompts_masks)

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.model.pad_token_id, -100)

        loss = torch.zeros([], device=self.device)

        for codebook in range(self.num_codebooks):
            logits_k = logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            loss += self.loss_fn[0](logits_k, targets_k)

        loss = loss / self.num_codebooks

        return loss

    @torch.no_grad()
    def test_model(self, batch):
        if not self.config.train.test_model:
            return

        _, prompts, prompts_masks, _ = batch

        tokens = self.generator.sampling(prompts, prompts_masks)
        y = self.audio_codec.decode(tokens)

        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.model.sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            y.write(f.name)
            self.logger.experiment.log(
                {
                    "audio": wandb.Audio(f.name, caption="audio"),
                }
            )

    def training_step(self, batch, batch_idx):
        if batch_idx % 10_000 == 0:
            self.test_model(batch)

        # if the batch is too small, skip it (I should do that in the pipeline)
        if batch[0].shape[-1] < (self.num_codebooks + 1):
            return

        loss = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True)
        self.loss_metric[0](loss)

        if not self.trainer.fit_loop._should_accumulate():
            self.log("Accumulate loss", self.loss_metric[0].compute())
            self.loss_metric[0].reset()

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx < 3:
            self.test_model(batch)

        loss = self.step(batch, batch_idx)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Warmup Scheduler
        warmup_steps = self.config.train.warmup_steps
        lr_max = self.config.train.lr_max
        lr_min = self.config.train.lr_min

        # bnb.optim.AdamW8bit optimizer
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1
        )

        def lr_lambda_warmup(current_step):
            return min(1.0, float(current_step) / float(warmup_steps))

        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup)

        # CosineAnnealingLR Scheduler
        total_steps = self.trainer.estimated_stepping_batches
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps), eta_min=lr_min
        )

        # Combine schedulers with SequentialLR
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
