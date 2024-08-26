import tempfile

import lightning as L
import torch
from audiotools import AudioSignal
from torch import optim
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
        self.model = WaveAI(self.config)
        self.save_hyperparameters()

        # put in list to avoid training in the pytorch-lightning (a bit hacky)
        self.loss_metric = [LossTensor()]
        self.loss_fn = [CrossEntropyLoss()]

        self.generator = Generation(self.model)
        self.audio_codec = audio_autoencoder()
        self.audio_codec.model.to("cpu")

        self.nbm_sample_gen = -1  # skip first run when lightning module is initialized

    def step(self, batch, batch_idx) -> torch.Tensor:
        tgt_audio, prompts, lyrics = batch

        tgt_audio = tgt_audio.to(self.device)
        prompts = prompts.to(self.device)
        lyrics = lyrics.to(self.device)

        # cut the audio to the max length
        tgt_audio = tgt_audio[:, :, : self.config.model.max_seq_length]

        # just for logging (to see the number of tokens)
        self.log("nbm_token", tgt_audio.numel())

        # get the delay pattern, in this way each token is delayed by the same amount of time
        inputs, _ = self.delay_pattern.build_delay_pattern_mask(
            tgt_audio, self.config.model.pad_token_id, self.config.model.max_seq_length
        )

        # shift the tokens to the right (like that the model will predict the next token and will not see the future)
        inputs = self.delay_pattern.shift_tokens_right(
            inputs, self.config.model.pad_token_id, self.config.model.pad_token_id
        ).to(self.device)

        # create the inputs and labels tensors
        inputs_ids = inputs[..., :-1]
        labels = inputs[..., 1:]

        logits = self.model(inputs_ids, prompts, lyrics)

        # get logits values without prepends embedding
        logits = logits[..., -labels.size(-1) :, :]

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.model.pad_token_id, -100)

        loss = torch.zeros([], device=self.device)

        for codebook in range(self.config.model.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            # get index of the most probable token
            # max_prob_idx = logits_k.argmax(dim=-1)
            # print(targets_k[..., 10:20], max_prob_idx[..., 10:20])

            loss += self.loss_fn[0](logits_k, targets_k)

        loss = loss / self.config.model.num_codebooks

        return loss

    def training_step(self, batch, batch_idx):
        # reset the counter for the eval step experiment
        self.nbm_sample_gen = 0

        # if the batch is too small, skip it (I should do that in the pipeline)
        if batch[0].shape[-1] < (self.config.model.num_codebooks + 1):
            return

        loss = self.step(batch, batch_idx)

        self.log("train_loss", loss, on_step=True)
        self.loss_metric[0](loss)

        if not self.trainer.fit_loop._should_accumulate():
            self.log("Accumulate loss", self.loss_metric[0].compute())
            self.loss_metric[0].reset()

        return loss

    def validation_step(self, batch, batch_idx):
        if self.nbm_sample_gen < 4 and self.nbm_sample_gen != -1:
            # run experiment on a single sample from batch
            prompts = batch[1][0].unsqueeze(0)
            lyrics = batch[2][0].unsqueeze(0)

            prompts = prompts.to(self.device)
            lyrics = lyrics.to(self.device)

            output_ids = self.generator.sampling(prompts, lyrics)
            with torch.no_grad():
                y = self.audio_codec.decode(output_ids)

            y = AudioSignal(
                y.cpu().numpy(), sample_rate=self.audio_codec.model.sample_rate
            )

            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                y.write(f.name)
                self.logger.experiment.log(
                    {
                        "audio": wandb.Audio(f.name, caption="audio"),
                    }
                )

            self.nbm_sample_gen += 1

        loss = self.step(batch, batch_idx)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Warmup Scheduler
        warmup_steps = self.config.train.warmup_steps
        lr_max = self.config.train.lr_max
        lr_min = self.config.train.lr_min

        optimizer = optim.AdamW(
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
