"""
Delay Pattern Module
"""

import torch


class DelayPattern:
    """
    Class to build a delayed pattern mask for the input_ids.
    This is used to predict the next token in the sequence
    """

    @staticmethod
    def build_delay_pattern_mask(
        input_ids: torch.LongTensor, pad_token_id: int, max_seq_length: int
    ):
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        b, k, seq_len = input_ids.shape

        delays_ids = torch.full(
            (b, k, max_seq_length + (k - 1)), pad_token_id, dtype=torch.long
        )

        for k_idx in range(k):
            delays_ids[:, k_idx, k_idx : max_seq_length + k_idx] = torch.full_like(
                delays_ids[:, k_idx, k_idx : max_seq_length + k_idx], -1
            )

        for k_idx in range(k):
            delays_ids[:, k_idx, k_idx : seq_len + k_idx] = input_ids[:, k_idx, :]

        mask = torch.where(delays_ids == pad_token_id, pad_token_id, -1)
        return delays_ids[..., :seq_len], mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask.
        """
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]

        input_ids = torch.where(
            decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask
        )
        return input_ids

    @staticmethod
    def shift_tokens_right(
        inputs_ids_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ):
        """
        Shift input ids one token to the right.
        """
        # transpose to get (bsz, num_codebooks, seq_len)
        # inputs_ids_ids = inputs_ids_ids.transpose(1, 2)
        shifted_inputs_ids_ids = inputs_ids_ids.new_zeros(inputs_ids_ids.shape)
        shifted_inputs_ids_ids[..., 1:] = inputs_ids_ids[..., :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError(
                "Make sure to set the decoder_start_token_id attribute of the model's configuration."
            )
        shifted_inputs_ids_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError(
                "Make sure to set the pad_token_id attribute of the model's configuration."
            )
        # replace possible -100 values in labels by `pad_token_id`
        shifted_inputs_ids_ids.masked_fill_(
            shifted_inputs_ids_ids == -100, pad_token_id
        )

        return shifted_inputs_ids_ids

    def reverse_delay_pattern_mask(self, input_ids):
        """Reverse the delay pattern mask to the input_ids. This is used to predict the next token in the sequence"""
        b, k, seq_len = input_ids.shape

        delays_ids = torch.full((b, k, seq_len - (k - 1)), 0, dtype=torch.long)

        for k_idx in range(k):
            delays_ids[:, k_idx, :] = input_ids[
                :, k_idx, k_idx : seq_len - (k - 1) + k_idx
            ]

        return delays_ids
