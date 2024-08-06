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
        padding_mask = batch[1]

        if self.config.cross_att:
            src_text = batch[2]

        decoder_input_ids_start = self.model.prepare_inputs_for_generation(
            batch[0].size(0)
        ).to(tgt_audio.device)

        tgt_audio = torch.cat((decoder_input_ids_start, tgt_audio), dim=-1)

        # cut the audio to the max length (including the codebooks because of the delay pattern)
        tgt_audio = tgt_audio[
            :, :, : self.config.max_seq_length - self.config.num_codebooks
        ]

        padding_mask = torch.cat(
            (torch.ones_like(decoder_input_ids_start)[:, 0], padding_mask),
            dim=-1,
        )

        padding_mask = padding_mask[..., : tgt_audio.size(-1)]

        inputs_id, _ = self.delay_pattern.build_delay_pattern_mask(
            tgt_audio,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        labels = inputs_id.detach().clone()

        logits = self.model(
            inputs_id, padding_mask, src_text if self.config.cross_att else None
        )

        loss = torch.zeros([])

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss_fn = CrossEntropyLoss()
        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            # get index of the most probable token
            #  max_prob_idx = logits_k.argmax(dim=-1)

            # print(targets_k, max_prob_idx)

            loss += loss_fn(logits_k.cpu(), targets_k.cpu())

        loss = loss / self.config.num_codebooks

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tgt_audio = batch[0]
        padding_mask = batch[1]

        if self.config.cross_att:
            src_text = batch[1]

        # cut the audio to the max length (including the codebooks because of the delay pattern)
        tgt_audio = tgt_audio[
            :, :, : self.config.max_seq_length - self.config.num_codebooks + 1
        ]
        padding_mask = padding_mask[..., : tgt_audio.size(-1)]

        inputs_id, _ = self.delay_pattern.build_delay_pattern_mask(
            tgt_audio,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        labels = inputs_id.detach().clone()

        logits = self.model(
            inputs_id, padding_mask, src_text if self.config.cross_att else None
        )

        loss = torch.zeros([])

        # ignore the pad token (when pytorch see -100 in the labels it will ignore it)
        labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

        loss_fn = CrossEntropyLoss()
        for codebook in range(self.config.num_codebooks):
            logits_k = (
                logits[:, codebook, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, prob]
            targets_k = labels[:, codebook, ...].contiguous().view(-1)  # [B x T]

            loss += loss_fn(logits_k.cpu(), targets_k.cpu())

        loss = loss / self.config.num_codebooks

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1
        )
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6
        # )
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=1e-6,
            total_iters=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "val_loss"}
        ]
