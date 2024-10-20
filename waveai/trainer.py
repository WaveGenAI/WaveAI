import bitsandbytes as bnb
import lightning as L
import torch
from audiotools import AudioSignal
from lightning.pytorch.utilities import grad_norm
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler

from .audio_processor import AudioProcessor
from .generation import Generation
from .model import WaveAI
from .utils.config_parser import ConfigParser
from .utils.logs import LossTensor
from .utils.pattern import DelayPattern
from .utils.utils import gaussian_noise_gen


class Trainer(L.LightningModule):
    def __init__(self, config: ConfigParser, audio_processor: AudioProcessor = None):
        """Initialize the Trainer class.

        Args:
            config (ConfigParser): the configuration
            audio_processor (AudioProcessor): the audio processor
        """

        super().__init__()
        self.config = config

        self.delay_pattern = DelayPattern(self.config.model.stereo)

        self.model = WaveAI(
            self.config.model.num_codebooks,
            self.config.model.codebook_size,
            self.config.model.hidden_size,
            self.config.model.decoder_depth,
            self.config.model.decoder_heads,
            self.config.model.memory_dim,
            self.config.model.rotary_emb,
        )

        if self.config.model.compile:
            self.model = torch.compile(self.model)

        # self.model.load_pretrained(self.device) -> also I have to disable weight initialization
        self.save_hyperparameters(config.__dict__)

        # put in list to avoid training in the pytorch-lightning (a bit hacky)
        self.loss_metric = [LossTensor()]
        self.loss_fn = [CrossEntropyLoss()]

        self.audio_processor = audio_processor

        self.generator = Generation(
            self.model,
            self.config.model.num_codebooks,
            self.config.model.pad_token_id,
            self.config.model.stereo,
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the model.

        Args:
            logits (torch.Tensor): the logits predicted by the model
            labels (torch.Tensor): the labels

        Returns:
            torch.Tensor: the cross entropy loss between the logits and the labels
        """

        loss = torch.zeros([], device=self.device)

        for codebook in range(self.config.model.num_codebooks):
            logits_k = logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            loss += self.loss_fn[0](logits_k, targets_k)

        loss = loss / self.config.model.num_codebooks
        return loss

    def step(self, batch, batch_idx) -> torch.Tensor:
        input_ids, prompts, prompts_masks, *_ = batch
        # TODO: manage padding mask

        # just for logging (to see the number of tokens)
        self.log("nbm_token", input_ids.numel())

        # get the delay pattern, in this way each token is delayed by the same amount of time
        input_ids, delay_pattern_mask = self.delay_pattern.build_delay_pattern_mask(
            input_ids, self.config.model.pad_token_id, self.config.model.max_seq_length
        )

        inputs_ids = self.delay_pattern.apply_delay_pattern_mask(
            input_ids, delay_pattern_mask
        )

        # shift the tokens to the right (like that the model will predict the next token and will not see the future)
        input_ids = self.delay_pattern.shift_tokens_right(
            input_ids, self.config.model.pad_token_id, self.config.model.pad_token_id
        )

        # create the inputs and labels tensors
        inputs_ids = input_ids[..., :-1]
        labels = input_ids[..., 1:]

        # add random noise to the inputs
        inputs_ids = gaussian_noise_gen(
            inputs_ids,
            self.config.train.noise_mean,
            self.config.train.noise_std,
            max_val=self.config.model.pad_token_id,
            ignore_token=[self.config.model.pad_token_id, -100],
        )

        logits = self.model(inputs_ids, prompts, prompts_masks)

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.model.pad_token_id, -100)

        return self.compute_loss(logits, labels)

    @torch.no_grad()
    def test_model(self, batch) -> AudioSignal:
        if not self.config.train.test_model:
            return

        _, prompts, prompts_masks, *_ = batch

        # get first value from batch
        prompts = prompts[0, ...].unsqueeze(0)
        prompts_masks = prompts_masks[0, ...].unsqueeze(0)

        tokens = self.generator.sampling(prompts, prompts_masks)
        audio = self.audio_processor.decode_audio(tokens)

        return audio

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Training step of the model.

        Args:
            batch (torch.Tensor): the batch
            batch_idx (int): the batch index

        Returns:
            dict: the loss and the predictions made by the model
        """

        # if the batch is too small, skip it (I should do that in the pipeline)
        if batch[0].shape[-1] < (self.config.model.num_codebooks + 1):
            return

        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True)
        self.loss_metric[0](loss)

        # log the batch loss
        if batch_idx % self.config.train.accumulate_grad_batches == 0:
            self.log("Batch loss", self.loss_metric[0].compute())
            self.loss_metric[0].reset()

        y = None
        if batch_idx % self.config.train.log_every_n_steps == 0:
            y = self.test_model(batch)

        return {"loss": loss, "predictions": y}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Validation step of the model.

        Args:
            batch (torch.Tensor): the batch
            batch_idx (int): the batch index

        Returns:
            dict: the loss and the predictions made by the model
        """

        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)

        y = None
        if batch_idx < 10:
            y = self.test_model(batch)

        return {"loss": loss, "predictions": y}

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

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
