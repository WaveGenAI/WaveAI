import torch
import torch.nn.functional as F

from .utils.pattern import DelayPattern


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


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
        # tokens: [batch_size, channel * num_codebooks, seq_length]
        tokens = (
            torch.ones(1, self.num_codebooks, 1, device=prompt.device).long()
            * self.pad_token
        )
        step = duration * 86 + (self.num_codebooks - 1)
        tokens, padding_mask = self.pattern.build_delay_pattern_mask(
            tokens, self.pad_token, step
        )

        b, k, _ = tokens.size()
        with torch.no_grad():
            for i in range(step):
                logits = self.model.forward(tokens, prompt, prompt_padding_mask)
                logits = logits[..., -1, :]  # get the last token
                logits = logits.view(-1, logits.size(-1))  # flatten the logits

                # Sample the next token
                out = topk_sampling(logits, top_k=top_k, temperature=temperature)
                out = out.view(
                    b, k, 1
                )  # reshape the output to [batch_size, num_codebooks, 1]

                tokens = torch.cat([tokens, out], dim=-1)

                print(f"Step {i + 1} / {step}", end="\r")

        tokens = self.pattern.reverse_delay_pattern_mask(tokens, padding_mask)[..., 1:]

        if self.stereo:
            # convert 1 x (num_codebooks x channels) x seq_length to 1 x channels x num_codebooks x seq_length
            tokens = tokens.view(1, 2, self.num_codebooks // 2, -1)

            # remove the batch dimension
            tokens = tokens.squeeze(0)

        return tokens
