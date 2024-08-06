import torch
import torch.nn.functional as F

from model.pattern import DelayPattern


class Generation:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.pattern = DelayPattern(self.model.config.num_codebooks)

    def beam_search(self, *args, **kwargs):
        raise NotImplementedError

    def sampling(
        self,
        cross_att_emb: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1,
        top_k: int = 250,
    ) -> torch.Tensor:

        _, mask = self.pattern.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.model.config.pad_token_id,
            max_length=self.model.config.max_seq_length,
        )

        output_ids = self.model.prepare_inputs_for_generation().to(mask.device)

        steps = self.model.config.max_seq_length - output_ids.size(-1)

        for i in range(steps):
            padding_mask = torch.ones_like(output_ids)[:, 0, :]

            inputs_ids_pred = self.pattern.apply_delay_pattern_mask(output_ids, mask)

            logits = self.model(inputs_ids_pred, padding_mask, cross_att_emb)
            last_toks_logits = logits[
                :, :, -1, :
            ]  # shape: [batch_size, num_heads, vocab_size]

            v, _ = torch.topk(last_toks_logits, min(top_k, logits.size(-1)))

            last_toks_logits[last_toks_logits < v[..., -1, None]] = float("-inf")

            probs = F.softmax(last_toks_logits, dim=-1)

            probs = (probs / temperature).squeeze(0)

            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)

            output_ids = torch.cat((output_ids, item_next[None, ...]), dim=-1)

            print(f"Step {i + 1} / {steps}", end="\r")

        output_ids = self.pattern.apply_delay_pattern_mask(output_ids, mask)

        return output_ids
