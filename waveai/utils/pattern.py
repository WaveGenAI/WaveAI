"""
Delay Pattern Module
"""

import typing

import torch


class DelayPattern:
    """
    Class to build a delayed pattern mask for the input_ids.
    This is used to predict the next token in the sequence
    """

    def __init__(self, stereo: bool = False):
        """Initialize the DelayPattern

        Args:
            stereo (bool, optional): Apply stereo pattern. Defaults to False.
        """

        self._stereo = stereo

    def build_delay_pattern_mask(
        self, input_ids: torch.LongTensor, pad_token_id: int, max_seq_length: int
    ) -> typing.Tuple[torch.LongTensor, torch.LongTensor]:
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one token (not when stereo).

        Args:
            input_ids (torch.LongTensor): the input ids
            pad_token_id (int): the pad token id
            max_seq_length (int): the maximum sequence length

        Returns:
            typing.Tuple[torch.LongTensor, torch.LongTensor]: the delayed pattern mask and the padding mask
        """
        if self._stereo:
            input_ids = self._stereo_convert(input_ids)

        b, k, seq_len = input_ids.shape

        delays_ids = torch.full(
            (b, k, max_seq_length),
            pad_token_id,
            dtype=torch.long,
        )
        delays_ids = delays_ids.to(input_ids)

        if not self._stereo:
            # Create the mono pattern like
            # [P, -1, -1, -1, -1, -1]
            # [P, P, -1, -1, -1, -1]
            # [P, P, P, -1, -1, -1]
            # etc.

            for k_idx in range(k):
                delays_ids[:, k_idx, k_idx : max_seq_length + k_idx] = torch.full_like(
                    delays_ids[:, k_idx, k_idx : max_seq_length + k_idx], -1
                )
        else:
            for col_idx, k_idx in enumerate(range(0, k, 2)):
                # Create the stereo pattern like
                # [P, -1, -1, -1, -1, P]
                # [P, -1, -1, -1, -1, P]
                # [P, P, -1, -1, -1, -1,]
                # [P, P, -1, -1, -1, -1]
                # etc.

                delays_ids[:, k_idx, col_idx : max_seq_length + col_idx] = (
                    torch.full_like(
                        delays_ids[:, k_idx, col_idx : max_seq_length + col_idx], -1
                    )
                )

                delays_ids[:, k_idx + 1, col_idx : max_seq_length + col_idx] = (
                    torch.full_like(
                        delays_ids[:, k_idx, col_idx : max_seq_length + col_idx], -1
                    )
                )

        for k_idx in range(k):
            id_start = torch.where(delays_ids[:, k_idx, :] == -1)[1][0]

            delays_ids[:, k_idx, id_start : min(seq_len + id_start, max_seq_length)] = (
                input_ids[:, k_idx, : max_seq_length - id_start]
            )

        mask = torch.where(delays_ids == pad_token_id, pad_token_id, -1)
        mask = mask.to(input_ids)
        return delays_ids[..., :seq_len], mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask.
        """
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]

        input_ids_pad = torch.where(
            decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask
        )
        input_ids_pad = input_ids_pad.to(input_ids)
        return input_ids_pad

    @staticmethod
    def _stereo_convert(codec: torch.Tensor) -> torch.Tensor:
        """Convert the codec tensor to stereo pattern

        Args:
            codec (torch.Tensor): the codec tensor

        Returns:
            torch.Tensor: the stereo codec tensor
        """

        out = torch.zeros_like(codec, dtype=codec.dtype)
        num_codebooks = codec.shape[1] // 2

        for i in range(num_codebooks):
            out[:, i * 2, :] = codec[:, i, :]
            out[:, i * 2 + 1, :] = codec[:, i + num_codebooks, :]

        return out

    @staticmethod
    def _stereo_unconvert(codec: torch.Tensor) -> torch.Tensor:
        """Convert the codec tensor to stereo pattern

        Args:
            codec (torch.Tensor): the codec tensor

        Returns:
            torch.Tensor: the stereo codec tensor
        """

        out = torch.zeros_like(codec, dtype=codec.dtype)
        num_codebooks = codec.shape[1] // 2

        for i in range(num_codebooks):
            out[:, i, :] = codec[:, i * 2, :]
            out[:, i + num_codebooks, :] = codec[:, i * 2 + 1, :]

        return out

    def reverse_delay_pattern_mask(
        self, input_ids: torch.Tensor, padding_maks: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse the delay pattern mask to the input_ids. This is used to predict the next token in the sequence

        Args:
            input_ids (torch.Tensor): the input ids
            padding_maks (torch.Tensor): the padding mask

        Returns:
            torch.Tensor: the reversed delay pattern mask
        """

        output = []

        _, k, _ = input_ids.shape

        # for each codebook, get the first -1 token and append the rest of the sequence according to that
        for k_idx in range(k):
            first_id = torch.where(padding_maks[:, k_idx, :] == -1)[1][0]
            output.append(input_ids[:, k_idx, first_id:])

        # get the minimum length of the output
        min_length = min([t.shape[-1] for t in output])
        delays_ids = [t[..., :min_length] for t in output]

        delays_ids = torch.stack(delays_ids, dim=1)

        if self._stereo:
            delays_ids = self._stereo_unconvert(delays_ids)

        return delays_ids
