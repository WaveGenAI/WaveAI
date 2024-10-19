import random

import torch


def audio_format_converter(codec: torch.Tensor, to_stereo: bool) -> torch.Tensor:
    """Convert the codec tensor to stereo and add a channel dimension if needed.

    Args:
        codec (torch.Tensor): the codec tensor

    Returns:
        torch.Tensor: the codec tensor
    """

    # if the shape is 2D, add a channel dimension
    if len(codec.shape) == 2:
        codec = codec.unsqueeze(-1)

    # if the codec is mono and the model is stereo, duplicate the channel
    if codec.shape[-1] == 1 and to_stereo:
        codec = torch.cat([codec, codec], dim=-1)

    return codec


def convert_to_tensor(row: dict, column: str) -> dict:
    """Convert the row column to a tensor.

    Args:
        row (dict): the row
        column (str): the column to convert

    Returns:
        dict: the row
    """

    row[column] = torch.tensor(row[column])

    return row


def gaussian_noise_gen(
    input: torch.Tensor,
    m: float = 0.1,
    std: float = 0.5,
    min_val: int = 0,
    max_val: int = 1024,
    ignore_token: int = [-100],
) -> torch.Tensor:
    """Add gaussian noise to the input tensor.

    Args:
        input (torch.Tensor): the input tensor
        m (float, optional): the mean of the gaussian distribution. Defaults to 0.1.
        std (float, optional): the standard deviation of the gaussian distribution. Defaults to 0.5.
        min_val (int, optional): minimum value of the noise. Defaults to 0.
        max_val (int, optional): maximum value of the noise. Defaults to 1024.
        ignore_token (list, optional): the token to ignore. Defaults to [-100].
    Returns:
        torch.Tensor: the input tensor with gaussian noise
    """
    total_tokens = input.numel()
    # Compute the number of tokens to modify using a gaussian distribution
    num_to_modify = int(random.gauss(m * total_tokens, m * total_tokens * std))
    prob = max(min(num_to_modify, total_tokens), 0) / total_tokens

    # Create a copy of the input tensor
    output = input.clone()

    # Create a mask for tokens that are not in ignore_token
    mask = torch.ones_like(input, dtype=torch.bool)
    for token in ignore_token:
        mask &= input != token

    # Generate random values
    val = torch.randint(min_val, max_val, input.shape, device=input.device)

    # Generate a random mask with the same shape as input
    random_mask = torch.rand(input.shape, device=input.device) < prob

    # Apply the modifications only to non-ignore tokens
    output = torch.where(mask & random_mask, val, output)

    return output
