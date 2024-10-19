import torch
import torch.nn.functional as F

from .utils.pattern import DelayPattern


class Generation:
    """
    Generation class to generate audio samples from the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_codebooks: int,
        pad_token: int,
        stereo: bool = False,
    ) -> None:
        self.model = model
        self.pattern = DelayPattern(stereo)

        self.stereo = stereo
        self.num_codebooks = num_codebooks
        self.pad_token = pad_token

    def sampling(
        self,
        prompt: torch.Tensor,
        prompt_padding_mask: torch.Tensor,
        duration=10,
        temperature: float = 1,
        top_k: int = 150,
    ) -> torch.Tensor:
        """Sample audio from the model.

        Args:
            prompt (torch.Tensor): the prompt tensor
            prompt_padding_mask (torch.Tensor): the prompt padding mask
            duration (int, optional): duration of the audio in seconds. Defaults to 10.
            temperature (float, optional): temperature for the sampling. Defaults to 1.
            top_k (int, optional): top k tokens to sample from. Defaults to 150.

        Returns:
            torch.Tensor: the predicted audio tensor
        """
        # tokens: [batch_size, channel * num_codebooks, seq_length]
        tokens = torch.ones(1, self.num_codebooks, 1).long() * self.pad_token
        step = duration * 86 + (self.num_codebooks - 1)
        tokens, padding_mask = self.pattern.build_delay_pattern_mask(
            tokens, self.pad_token, step
        )

        with torch.no_grad():
            for i in range(step):
                tokens = self.pattern.apply_delay_pattern_mask(
                    tokens.cpu(), padding_mask.cpu()
                )

                tokens = tokens.to(prompt.device)
                prompt = prompt.to(prompt.device)
                prompt_padding_mask = prompt_padding_mask.to(prompt.device)

                logits = self.model.forward(tokens, prompt, prompt_padding_mask)
                topk_tokens, indices = logits[:, :, -1, :].topk(top_k, dim=-1)
                topk_tokens = F.softmax((topk_tokens / temperature), dim=-1)
                samples = torch.multinomial(topk_tokens.view((-1, top_k)), 1).view(
                    topk_tokens.shape[:-1] + (1,)
                )
                new_tokens = torch.gather(indices, dim=-1, index=samples)
                tokens = torch.cat([tokens, new_tokens], dim=2)

                print(f"Step {i + 1} / {step}", end="\r")

        tokens = self.pattern.reverse_delay_pattern_mask(tokens, padding_mask)[..., 1:]

        if self.stereo:
            # convert 1 x (num_codebooks x channels) x seq_length to 1 x channels x num_codebooks x seq_length
            tokens = tokens.view(1, 2, self.num_codebooks // 2, -1)

            # remove the batch dimension
            tokens = tokens.squeeze(0)

        return tokens
