from typing import Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn


class EncodecConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        causal=False,
        pad_mode="reflect",
    ):
        super().__init__()

        self.causal = causal
        self.pad_mode = pad_mode

        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=True,
            )
        )

    @staticmethod
    def _pad1d(
        x: torch.Tensor,
        paddings: Tuple[int, int],
        mode: str = "zero",
        value: float = 0.0,
    ):
        length = x.shape[-1]
        if not mode == "reflect":
            return F.pad(x, paddings, mode, value)

        max_pad = max(paddings[0], paddings[1])
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, x):
        padding_total = self.conv.kernel_size[0] - self.conv.stride[0]
        padding_right = 0 if self.causal else padding_total // 2
        padding_left = padding_total if self.causal else padding_total - padding_right
        x = self._pad1d(x, (padding_left, padding_right), mode=self.pad_mode)
        x = self.conv(x)
        return x


class EncodecLSTM(nn.Module):
    def __init__(self, dimension, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0] + x
        x = x.permute(1, 2, 0)
        return x


class EncodecConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        causal=False,
        pad_mode="reflect",
    ):
        super().__init__()

        self.causal = causal
        self.pad_mode = pad_mode

        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=True,
            )
        )

    def forward(self, x):
        x = self.conv(x)

        # unpad, because it's a transpose convolution
        padding_total = self.conv.kernel_size[0] - self.conv.stride[0]
        padding_right = padding_total // (2 if not self.causal else 1)
        padding_left = padding_total - padding_right
        x = x[..., padding_left:-padding_right]
        return x


class EncodecResnetBlock(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.block = nn.ModuleList(
            [
                nn.ELU(alpha=1.0),
                EncodecConv1d(input_dim, input_dim // 2, kernel_size=(3,), stride=(1,)),
                nn.ELU(alpha=1.0),
                EncodecConv1d(input_dim // 2, input_dim, kernel_size=(1,), stride=(1,)),
            ]
        )

    def forward(self, x):
        x_ = x
        for layer in self.block:
            x = layer(x)
        return x + x_


class EncodecDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncodecConv1d(128, 1024, kernel_size=(7,), stride=(1,)),
                EncodecLSTM(dimension=1024, num_layers=2),
                nn.ELU(alpha=1.0),
                EncodecConvTranspose1d(1024, 512, kernel_size=(16,), stride=(8,)),
                EncodecResnetBlock(512),
                nn.ELU(alpha=1.0),
                EncodecConvTranspose1d(512, 256, kernel_size=(10,), stride=(5,)),
                EncodecResnetBlock(256),
                nn.ELU(alpha=1.0),
                EncodecConvTranspose1d(256, 128, kernel_size=(8,), stride=(4,)),
                EncodecResnetBlock(128),
                nn.ELU(alpha=1.0),
                EncodecConvTranspose1d(128, 64, kernel_size=(8,), stride=(4,)),
                EncodecResnetBlock(64),
                nn.ELU(alpha=1.0),
                EncodecConv1d(64, 1, kernel_size=(7,), stride=(1,)),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncodecEuclideanCodebook(nn.Module):
    def __init__(self, codebook_size=2048, codebook_dim=128):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(codebook_size, codebook_dim))

    def decode(self, tokens):
        return F.embedding(tokens, self.embed)


class EncodecVectorQuantization(nn.Module):
    def __init__(self, codebook_size=2048, codebook_dim=128):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(
            codebook_size=codebook_size, codebook_dim=codebook_dim
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)
        embed_in = self.codebook.encode(x)
        return embed_in

    def decode(self, tokens):
        x = self.codebook.decode(tokens)
        x = x.permute(0, 2, 1)
        return x


class EncodecQuantizer(nn.Module):
    def __init__(self, codebook_size=2048, frame_rate=50, num_quantizers=4):
        super().__init__()
        self.codebook_size = codebook_size
        self.frame_rate = frame_rate
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [
                EncodecVectorQuantization(codebook_size=codebook_size, codebook_dim=128)
                for _ in range(num_quantizers)
            ]
        )

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        x = torch.tensor(0.0, device=q.device)
        for codebook, tokens in zip(self.layers, q):
            vector = codebook.decode(tokens)
            x = x + vector
        return x


class EncodecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = EncodecDecoder()
        self.quantizer = EncodecQuantizer()

        self.device = torch.device("cpu")

    @torch.no_grad()
    def decode(self, q):
        q = q.transpose(0, 1)
        x = self.quantizer.decode(q)
        return self.decoder(x)

    def load_pretrained(self, device, model="facebook/encodec_32khz"):
        path = hf_hub_download(
            repo_id=model, filename="pytorch_model.bin", cache_dir=None
        )
        _values = torch.load(path, map_location=device)
        state_dict = {k: v for k, v in _values.items() if k in self.state_dict()}
        self.load_state_dict(state_dict)

        self.device = device


class Encodec:
    def __init__(self):
        self.model = EncodecModel()
        self.model.eval()

        self.sample_rate = 32000

    @torch.no_grad()
    def decode(self, q: torch.Tensor) -> torch.Tensor:
        """The decode function.

        Args:
            q (torch.Tensor): the codes to decode.

        Returns:
            torch.Tensor: the decoded audio signal.
        """
        q = q.to(self.model.device)
        q = q.transpose(0, 1)
        x = self.model.quantizer.decode(q)
        return self.model.decoder(x)

    def load_pretrained(self, device):
        self.model.load_pretrained(device)
