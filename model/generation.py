import torch
import torch.nn.functional as F


class Generation:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def beam_search(self, *args, **kwargs):
        raise NotImplementedError

    def sampling(
        self,
        inputs_id: torch.Tensor,
        mask: torch.Tensor,
        cross_att_emb: torch.Tensor,
        temperature: float = 1,
        top_k: int = 250,
    ) -> torch.Tensor:
        output_ids = inputs_id.clone()
        steps = self.model.config.max_seq_length - self.model.config.num_codebooks

        def top_k_sampling(logits, k):
            top_k_logits, _ = torch.topk(logits, k, dim=-1)
            indices_to_remove = logits < top_k_logits[..., -1, None]
            logits[indices_to_remove] = float("-inf")
            return logits

        for i in range(steps):
            input_ids = output_ids.clone()
            logits = self.model(input_ids, cross_att_emb)
            next_token_logits = logits[
                :, :, -1, :
            ]  # shape: [batch_size, num_heads, vocab_size]

            next_token_logits = top_k_sampling(next_token_logits, top_k)

            next_token_logits = next_token_logits / temperature

            next_token_probs = F.softmax(next_token_logits, dim=-1)

            batch_size, num_heads, vocab_size = next_token_probs.shape
            next_token_probs_flat = next_token_probs.view(-1, vocab_size)

            next_tokens_flat = torch.multinomial(next_token_probs_flat, num_samples=1)

            # Reshaping pour correspondre à la forme originale
            next_tokens = next_tokens_flat.view(batch_size, num_heads, 1)

            output_ids = torch.cat((output_ids, next_tokens), dim=-1)
            output_ids = self.model.apply_delay_pattern_mask(output_ids, mask)
            print(f"Step {i + 1} / {steps}", end="\r")
        return output_ids
