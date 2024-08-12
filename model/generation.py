import torch
import torch.nn.functional as F

from model.pattern import DelayPattern


class Generation:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.pattern = DelayPattern()

    def beam_search(self, *args, **kwargs):
        raise NotImplementedError

    def sampling(
        self,
        cross_att_emb: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 1,
        top_k: int = 250,
    ) -> torch.Tensor:

        output_ids, mask = self.pattern.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.model.config.model.pad_token_id,
            max_seq_length=self.model.config.model.max_seq_length,
        )

        steps = min(1000, self.model.config.model.max_seq_length - output_ids.size(-1))

        for i in range(steps):
            inputs_ids_pred = self.pattern.apply_delay_pattern_mask(output_ids, mask)

            logits = self.model(inputs_ids_pred.to(self.device))
            last_toks_logits = logits[
                :, :, -1, :
            ]  # shape: [batch_size, num_heads, vocab_size]

            v, _ = torch.topk(last_toks_logits, min(top_k, logits.size(-1)))

            last_toks_logits[last_toks_logits < v[..., -1, None]] = float("-inf")

            probs = F.softmax(last_toks_logits, dim=-1)

            probs = (probs / temperature).squeeze(0)

            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)

            output_ids = torch.cat((output_ids, item_next[None, ...].cpu()), dim=-1)

            print(f"Step {i + 1} / {steps}", end="\r")

        output_ids = self.pattern.apply_delay_pattern_mask(output_ids, mask)
        output_ids = self.pattern.reverse_delay_pattern_mask(output_ids)

        return output_ids
