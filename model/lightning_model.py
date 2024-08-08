import lightning as L
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler

from .config import Config
from .model import WaveAI
from .pattern import DelayPattern


class WaveAILightning(L.LightningModule):
    def __init__(self) -> None:
        """Lightning module for WaveAI.

        Args:
            use_prompt (bool, optional): is model trained on prompt. Defaults to False.
        """

        super().__init__()

        self.config = Config()
        self.delay_pattern = DelayPattern(self.config.num_codebooks)
        self.model = WaveAI(self.config)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        tgt_audio = batch[0]
        src_text = batch[1]

        if self.config.cross_att:
            src_text = src_text.to(self.device)

        tgt_audio = tgt_audio.to(self.device)

        # cut the audio to the max length (including the codebooks because of the delay pattern)
        tgt_audio = tgt_audio[:, :, : self.config.max_seq_length]

        input_ids, delay_pattern_mask = self.delay_pattern.build_delay_pattern_mask(
            tgt_audio,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length + self.config.num_codebooks,
        )
        input_ids = self.delay_pattern.apply_delay_pattern_mask(
            input_ids, delay_pattern_mask
        )

        logits = self.model(input_ids, src_text)

        labels = self.model.shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.pad_token_id
        )

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss = torch.zeros([], device=self.device)
        loss_fn = CrossEntropyLoss()

        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            # get index of the most probable token
            # max_prob_idx = logits_k.argmax(dim=-1)
            # print(targets_k[..., 10:20], max_prob_idx[..., 10:20])

            loss += loss_fn(logits_k, targets_k)

        loss = loss / self.config.num_codebooks

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tgt_audio = batch[0]
        src_text = batch[1]

        if self.config.cross_att:
            src_text = src_text.to(self.device)

        tgt_audio = tgt_audio.to(self.device)

        # cut the audio to the max length (including the codebooks because of the delay pattern)
        tgt_audio = tgt_audio[:, :, : self.config.max_seq_length]

        input_ids, delay_pattern_mask = self.delay_pattern.build_delay_pattern_mask(
            tgt_audio,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length + self.config.num_codebooks,
        )
        input_ids = self.delay_pattern.apply_delay_pattern_mask(
            input_ids, delay_pattern_mask
        )

        logits = self.model(input_ids, src_text)

        labels = self.model.shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.pad_token_id
        )

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss = torch.zeros([], device=self.device)
        loss_fn = CrossEntropyLoss()

        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            # get index of the most probable token
            # max_prob_idx = logits_k.argmax(dim=-1)
            # print(targets_k[..., 10:20], max_prob_idx[..., 10:20])

            loss += loss_fn(logits_k, targets_k)

        loss = loss / self.config.num_codebooks

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "val_loss"}
        ]
