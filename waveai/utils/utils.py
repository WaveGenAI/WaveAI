# some codes comes from https://github.com/jasonppy/VoiceCraft/blob/master/models/modules/utils.py
import random

import torch
import webdataset as wds
from datasets import load_dataset


def audio_format_converter(codec: torch.Tensor, to_stereo: bool) -> torch.Tensor:
    """Convert the codec tensor to stereo and add a channel dimension if needed.

    Args:
        codec (torch.Tensor): the codec tensor of shape (opt channel, num_codebooks, seq_length)

    Returns:
        torch.Tensor: the codec tensor of shape (channel, num_codebooks, seq_length)
    """

    # if the shape is 2D, add a channel dimension
    if len(codec.shape) == 2:
        codec = codec.unsqueeze(0)

    # if the codec is mono and the model is stereo, duplicate the channel
    if codec.shape[0] == 1 and to_stereo:
        codec = torch.cat([codec, codec], dim=0)

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


def shift_tokens_right(
    inputs_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_inputs_ids = inputs_ids.new_zeros(inputs_ids.shape)
    shifted_inputs_ids = shifted_inputs_ids.to(inputs_ids)

    shifted_inputs_ids[..., 1:] = inputs_ids[..., :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_inputs_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_inputs_ids.masked_fill_(shifted_inputs_ids == -100, pad_token_id)

    return shifted_inputs_ids


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths < lengths.unsqueeze(-1)


def load_webdataset(
    dataset_name: str, split_name: str, map: callable = None, shuffle: bool = True
) -> wds.WebDataset:
    """Load a webdataset dataset.

    Args:
        dataset_name (str): the name of the dataset
        split_name (str): the name of the split
        map (callable): the map function. Defaults to None.
        shuffle (bool, optional): shuffle the datase. Defaults to True.

    Returns:
        wds.WebDataset: the webdataset dataset
    """
    dataset = load_dataset(dataset_name, streaming=True)
    org, dataset_name = dataset_name.split("/")
    n_shards = dataset[split_name].n_shards

    url = f"/media/works/data/data/{split_name}-{{000000..{n_shards - 1}}}.tar"
    # url = f"pipe:curl --connect-timeout 60 --retry 50 --retry-delay 10 --retry-all-errors -f -s -L {url} -H 'Authorization:Bearer {get_token()}'"

    if shuffle:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.detshuffle(),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(map),
        )
    else:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(map),
        )

    return dataset
