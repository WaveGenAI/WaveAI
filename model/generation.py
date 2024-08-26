import torch
import torch.nn.functional as F

from model.pattern import DelayPattern


class Generation:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.pattern = DelayPattern()

    def beam_search(self, *args, **kwargs):
        raise NotImplementedError

    def sampling(
        self,
        cross_att_emb: torch.Tensor,
        prepends_ids: torch.Tensor,
        temperature: float = 1,
        top_k: int = 250,
    ) -> torch.Tensor:

        num_codebooks = self.model.config.model.num_codebooks
        if self.model.config.model.stereo:
            num_codebooks = 2 * num_codebooks

        input_ids = torch.ones((1, num_codebooks, 1))
        input_ids += self.model.config.model.pad_token_id - 1

        output_ids, mask = self.pattern.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.model.config.model.pad_token_id,
            max_seq_length=self.model.config.model.max_seq_length,
        )

        steps = min(
            1000,
            self.model.config.model.max_seq_length
            - output_ids.size(-1)
            - prepends_ids.size(-1),
        )

        for i in range(steps):
            inputs_ids_pred = self.pattern.apply_delay_pattern_mask(output_ids, mask)
            inputs_ids_pred = inputs_ids_pred.to(prepends_ids.device)

            # convert prepends_ids to long and input_ids to long
            inputs_ids_pred = inputs_ids_pred.long()
            prepends_ids = prepends_ids.long()

            logits = self.model(inputs_ids_pred, cross_att_emb, prepends_ids)

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
        output_ids = self.pattern.reverse_delay_pattern_mask(output_ids)[..., 1:]

        if self.model.config.model.stereo:
            # convert 1 x (num_codebooks x channels) x seq_length to 2 x num_codebooks x seq_length
            output_ids = output_ids.view(2, self.model.config.model.num_codebooks, -1)

        return output_ids
