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

    def inference(
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

        inputs = (
            torch.ones(1, self.num_codebooks, 1, device=prompt.device).long()
            * self.pad_token
        )

        step = duration * 86 + (self.num_codebooks - 1)
        inputs, padding_mask = self.pattern.build_delay_pattern_mask(
            inputs, self.pad_token, inputs.shape[-1] + step
        )

        # padding mask is usefull to get the index where the prediction starts and remove the tokens
        # generated where padding is applied

        with torch.no_grad():
            for i in range(step):
                inputs = self.pattern.apply_delay_pattern_mask(inputs, padding_mask)

                # create padding mask
                inputs_mask = torch.ones(
                    inputs.shape[0],
                    inputs.shape[2],
                    device=inputs.device,
                ).bool()

                # generate the next token
                logits, _ = self.model.forward(
                    inputs, inputs_mask, prompt, prompt_padding_mask
                )

                logits = logits[:, :, -1, :]  # b, k, seq_length, vocab_size

                # apply temperature
                logits = logits / temperature

                # convert logits to probabilities
                logits = F.softmax(logits, dim=-1)

                # get the top k largest tokens of shape b, k, top_k
                topk_tokens, indices = logits.topk(top_k, dim=-1)

                # convert to 2D tensor
                topk_tokens = topk_tokens.view(-1, top_k)

                # sample from the top k tokens, it returns the indices of the sampled tokens
                samples = torch.multinomial(topk_tokens, 1).unsqueeze(0)

                # get the indices of the sampled tokens
                indexs = torch.gather(indices, -1, samples)

                # concatenate the sampled tokens to the inputs
                inputs = torch.cat([inputs, indexs], dim=2)
                print(f"Step {i + 1} / {step}", end="\r")

        outputs = self.pattern.reverse_delay_pattern_mask(inputs, padding_mask)[..., 1:]

        if self.stereo:
            # convert 1 x (num_codebooks x channels) x seq_length to 1 x channels x num_codebooks x seq_length
            outputs = outputs.view(1, 2, self.num_codebooks // 2, -1)

            # remove the batch dimension
            outputs = outputs.squeeze(0)

        # TODO: change that
        # replace the pad tokens with 0
        outputs = torch.where(outputs == self.pad_token, 0, outputs)

        return outputs
