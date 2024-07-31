""" 
Class that implements multiple generation methods.
"""

import torch
import torch.nn.functional as F


class Generation:
    """
    Generation class that implements multiple generation
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def beam_search(self, *args, **kwargs):
        """Beam search"""
        raise NotImplementedError

    def greedy_decoding(
        self,
        inputs_id: torch.Tensor,
        mask: torch.Tensor,
        cross_att_emb: torch.Tensor,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Greedy decoding
        Args:
        inputs_id (torch.Tensor): the input tensor
        mask (torch.Tensor): the mask tensor
        cross_att_emb (torch.Tensor): the cross attention embedding
        repetition_penalty (float, optional): penalty factor for repeated tokens. Defaults to 1.2.
        Returns:
        torch.Tensor: the output tensor
        """
        output_ids = inputs_id.clone()
        steps = self.model.config.max_seq_length - self.model.config.num_codebooks

        def apply_repeat_penalty(logits, context, penalty):
            """Apply repetition penalty"""

            # TODO: Implement repetition penalty

            return logits

        for i in range(steps):
            input_ids = output_ids.clone()
            logits = self.model(input_ids, cross_att_emb)
            next_token_logits = logits[:, :, -1, :]  # get the last token logits

            # Apply repetition penalty
            next_token_logits = apply_repeat_penalty(
                next_token_logits, input_ids, repetition_penalty
            )

            # Calculate probabilities
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_tokens = torch.argmax(next_token_probs, dim=-1)
            next_tokens = next_tokens.view(
                next_token_probs.size(0), next_token_probs.size(1), 1
            )
            output_ids = torch.cat((output_ids, next_tokens), dim=-1)
            output_ids = self.model.apply_delay_pattern_mask(output_ids, mask)
            print(f"Step {i + 1} / {steps}", end="\r")

        return output_ids
