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
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        duration=10,
        temperature: float = 1,
        top_k: int = 150,
    ) -> torch.Tensor:
        # tokens: [batch_size, channel * num_codebooks, seq_length]
        tokens = (torch.ones(1, self.num_codebooks, 1).long() * self.pad_token).to(
            "cuda"
        )

        step = duration * 86 + (self.num_codebooks - 1)
        tokens, padding_mask = self.pattern.build_delay_pattern_mask(
            tokens, self.pad_token, step
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in range(step):
                    tokens = self.pattern.apply_delay_pattern_mask(tokens, padding_mask)
                    logits = self.model.forward(tokens, memory, memory_key_padding_mask)
                    topk_tokens, indices = logits[:, :, -1, :].topk(top_k, dim=-1)
                    topk_tokens = F.softmax((topk_tokens / temperature), dim=-1)
                    samples = torch.multinomial(topk_tokens.view((-1, top_k)), 1).view(
                        topk_tokens.shape[:-1] + (1,)
                    )
                    new_tokens = torch.gather(indices, dim=-1, index=samples)
                    tokens = torch.cat([tokens, new_tokens], dim=2)

                    print(f"Step {i + 1} / {step}", end="\r")

        tokens = torch.stack(
            [
                tokens[:, 0, 0:-3],
                tokens[:, 1, 1:-2],
                tokens[:, 2, 2:-1],
                tokens[:, 3, 3:],
            ],
            dim=1,
        )[:, :, 1:]

        if self.stereo:  # TODO: check this
            # convert 1 x (num_codebooks x channels) x seq_length to 2 x num_codebooks x seq_length
            tokens = tokens.view(1, 2, self.num_codebooks // 2, -1)

            # remove the batch dimension
            tokens = tokens.squeeze(0)

        return tokens
