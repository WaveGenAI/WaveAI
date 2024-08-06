"""
Delay Pattern Module
"""

import torch


class DelayPattern:
    """
    Class to build a delayed pattern mask for the input_ids.
    This is used to predict the next token in the sequence
    """

    def __init__(self, nq: int):
        self.nq = nq

    def build_delay_pattern_mask(
        self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int = None
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
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        input_ids = input_ids.reshape(-1, self.nq, input_ids.shape[-1])

        bsz, num_codebooks, seq_len = input_ids.shape

        input_ids_shifted = (
            torch.ones(
                (bsz, num_codebooks, max_length),
                dtype=torch.long,
                device=input_ids.device,
            )
            * -1
        )

        channel_codebooks = num_codebooks
        # we only apply the mask if we have a large enough seq len - otherwise we return as is
        if max_length < 2 * channel_codebooks - 1:
            return input_ids.reshape(
                bsz * num_codebooks, -1
            ), input_ids_shifted.reshape(bsz * num_codebooks, -1)

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(channel_codebooks):
            # mono channel - loop over the codebooks one-by-one
            input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[
                :, codebook
            ]

        # construct a pattern mask that indicates the positions of padding tokens for each codebook
        # first fill the upper triangular part (the EOS padding)
        delay_pattern = torch.triu(
            torch.ones((channel_codebooks, max_length), dtype=torch.bool),
            diagonal=max_length - channel_codebooks + 1,
        )
        # then fill the lower triangular part (the BOS padding)
        delay_pattern = delay_pattern + torch.tril(
            torch.ones((channel_codebooks, max_length), dtype=torch.bool)
        )

        mask = ~delay_pattern.to(input_ids.device)
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id

        # find the first position to start generating - this is the first place we have the -1 token
        # and will always be in the first codebook (since it has no codebook offset)
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]

        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = seq_len

        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        pattern_mask = input_ids
        input_ids = input_ids[..., :first_start_id]
        return input_ids, pattern_mask

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
