from typing import List

import torch
from audiotools import AudioSignal

from waveai.audio_autoencoder import DAC
from waveai.text_encoder import T5EncoderBaseModel
from waveai.utils.utils import audio_format_converter, make_pad_mask

from .utils.pattern import DelayPattern


class AudioProcessor:
    """
    Class to process audio data
    """

    def __init__(self, config):
        self.text_enc = T5EncoderBaseModel(
            name=config.model.text_encoder, max_seq_len=config.model.max_prompt_length
        )
        self.delay_pattern = DelayPattern(
            config.model.stereo
        )  # if stereo, use stereo Partial Delay Pattern https://arxiv.org/pdf/2306.05284
        self.audio_codec = DAC()
        self.config = config

    def encode_prompt(self, prompt: list) -> tuple:
        prompt_emds, prompt_masks = self.text_enc(prompt)
        return prompt_emds, prompt_masks

    def prepare_audio(
        self, codec: list, chunk_id: int, num_chunks: int
    ) -> torch.Tensor:
        # convert list of list to tensor
        codec = torch.Tensor(codec)

        # be sure the tensor has the right shape (add channel dimension if needed)
        codebooks = audio_format_converter(codec, self.config.model.stereo)

        # convert to batch x (channel x num_codebooks) x seq_length
        codebooks = codebooks.view(-1, codebooks.size(-1)).unsqueeze(0)

        # add the start token
        if chunk_id == 0:
            start_token = torch.zeros(
                codebooks.shape[0], codebooks.shape[1], 1, dtype=codebooks.dtype
            ).fill_(self.config.model.start_token_id)

            codebooks = torch.cat([start_token, codebooks], dim=-1)

        # add the end token
        if chunk_id == (num_chunks - 1):
            end_token = torch.zeros(
                codebooks.shape[0], codebooks.shape[1], dtype=codebooks.dtype
            ).fill_(self.config.model.end_token_id)

            # because of the delay pattern, we can't give the full length of the codebooks
            #  [[1, 2, 3, 4, 5, 6],
            #   [P, 1, 2, 3, 4, 5],
            #   [P, P, 1, 2, 3, 4],
            #   [P, P, P, 1, 2, 3],
            #   [...]]
            #
            # so we need to add the end token at the right position (position 3 in this example)

            pos_last_token = self.config.model.num_codebooks

            # if stereo, divide by 2 because of the stereo delay pattern
            if self.config.model.stereo:
                pos_last_token //= 2

            codebooks[:, :, -pos_last_token] = end_token

        if codebooks.size(1) > self.config.model.num_codebooks:
            codebooks = codebooks[
                :, : self.config.model.num_codebooks, :
            ]  # truncate the codebooks (dangerous depending of the delay pattern)

        # get the delay pattern
        #  [[1, 2, 3, 4, 5, 5], -> channel 1
        #   [1, 2, 3, 4, 5, 5], -> channel 2
        #   [5, 1, 2, 3, 4, 5], -> channel 1
        #   [5, 1, 2, 3, 4, 5], -> channel 2
        #   [...]]
        # when stereo
        input_ids, _ = self.delay_pattern.build_delay_pattern_mask(
            codebooks, self.config.model.pad_token_id, self.config.model.max_seq_length
        )

        # remove unused batch dimension and convert to long tensor
        input_ids = input_ids.squeeze(0).long()

        return input_ids

    def _pad_codebooks(self, codebooks: List[torch.Tensor], pad_token) -> torch.Tensor:
        """Pad the codebooks to the same length.

        Args:
            codebooks (List[torch.Tensor]): the codebooks to pad of shape [(num_codebooks, seq_length)].
            pad_token (int): the pad token.

        Returns:
            torch.Tensor: the padded codebooks of shape (batch, num_codebooks, seq_length).
        """

        # revert k, seq_len to seq_len, k (see https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
        codebooks = [codebook.T for codebook in codebooks]
        # pad the codebooks
        padded_codebooks = torch.nn.utils.rnn.pad_sequence(
            codebooks, padding_value=pad_token
        )

        # permute to b, k, seq_len
        padded_codebooks = padded_codebooks.permute(1, 2, 0)

        pad_full_length = torch.zeros(
            padded_codebooks.size(0),
            padded_codebooks.size(1),
            self.config.model.max_seq_length,
        ).fill_(pad_token)

        pad_full_length[:, :, : padded_codebooks.size(2)] = padded_codebooks

        return pad_full_length.long()

    @torch.no_grad()
    def collate_fn(self, rows: list) -> tuple:
        """Collate function for processing batches.

        Args:
            rows (list): the rows to process.

        Returns:
            tuple: the processed rows.
        """

        prompt = [row[self.config.data.text_column] for row in rows]
        # apply the delay pattern to the audio
        inputs = [
            self.prepare_audio(
                row[self.config.data.audio_column],
                int(row[self.config.data.chunk_id]),
                int(row[self.config.data.num_chunks]),
            )
            for row in rows
        ]
        # get the size of the inputs (seq_length)
        inputs_size = torch.tensor([input.size(-1) for input in inputs])

        # create the padding mask
        padding_mask = make_pad_mask(inputs_size, self.config.model.max_seq_length)

        # Pad the codec tensors and stack them
        inputs = self._pad_codebooks(inputs, pad_token=self.config.model.pad_token_id)

        # encode the prompt
        prompts_embeds, prompts_masks = self.encode_prompt(prompt)

        # audioMAE embeds for cosine similarity
        labels_mae = [
            torch.Tensor(row[self.config.data.audio_embed_column]) for row in rows
        ]

        labels_mae = [label.view(label.shape[0], -1) for label in labels_mae]

        # stack the audio embeds
        labels_mae = torch.stack(labels_mae)

        # TODO: Maybe remove prompt from the return because it's a str and not a tensor (but currently used for logging)
        return inputs, padding_mask, prompts_embeds, prompts_masks, labels_mae, prompt

    @torch.no_grad()
    def decode_audio(self, codec: torch.Tensor) -> AudioSignal:
        """Decode the codec tensor to audio.

        Args:
            codec (torch.Tensor): the codec tensor

        Returns:
            AudioSignal: the audio signal
        """
        y = self.audio_codec.decode(codec.cpu())
        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.sample_rate)

        return y
