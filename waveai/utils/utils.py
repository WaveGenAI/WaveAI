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
