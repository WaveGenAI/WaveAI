import torch
import torch.nn.functional as F

from model.pattern import DelayPattern


class Generation:
    def __init__(
        self,
        model: torch.nn.Module,
        num_codebooks: int,
        pad_token: int,
        stereo: bool = False,
    ) -> None:
        self.model = model
        self.pattern = DelayPattern()

        self.stereo = stereo
        self.num_codebooks = num_codebooks
        self.pad_token = pad_token

    def beam_search(self, *args, **kwargs):
        raise NotImplementedError

    def sampling(
        self,
        duration=10,
        temperature: float = 1,
        top_k: int = 150,
    ) -> torch.Tensor:
        # tokens: [batch_size, channel * num_codebooks, seq_length]
        tokens = (
            torch.ones((1, self.num_codebooks, 1), dtype=torch.long) * self.pad_token
        ).to("cuda")

        step = duration * 86
        tokens, padding_mask = self.pattern.build_delay_pattern_mask(
            tokens, self.pad_token, step
        )

        for i in range(step):
            tokens = self.pattern.apply_delay_pattern_mask(tokens, padding_mask)
            logits = self.model(tokens)

            topk, indices = logits[:, :, -1, :].topk(top_k, dim=-1)
            topk = F.softmax((topk / temperature), dim=-1)
            samples = torch.multinomial(topk.view((-1, top_k)), 1).view(
                topk.shape[:-1] + (1,)
            )
            new_tokens = torch.gather(indices, dim=-1, index=samples)

            tokens = torch.cat([tokens, new_tokens.long()], dim=2)

            print(f"Step {i + 1} / {step}", end="\r")

        tokens = self.pattern.reverse_delay_pattern_mask(tokens)[..., 1:]

        if self.stereo:
            # convert 1 x (num_codebooks x channels) x seq_length to 2 x num_codebooks x seq_length
            tokens = tokens.view(1, 2, self.num_codebooks // 2, -1)

            # remove the batch dimension
            tokens = tokens.squeeze(0)

        return tokens
