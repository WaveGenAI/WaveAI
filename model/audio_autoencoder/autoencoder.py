"""
This module contains the abstract class for the autoencoder model.
"""

import abc

import torch


class AutoEncoder:
    """
    Class for the autoencoder model.
    """

    @abc.abstractmethod
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress the input tensor without keeping everying in memory.

        Args:
            x (torch.Tensor): audio input of shape [1, 1, T]

        Returns:
            torch.Tensor: codebook output of shape [B, K, T]

        B: batch size
        K: number of codebooks
        T: sequence length
        """

        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        """Decompress the input tensor without keeping everything in memory.

        Args:
            z (torch.Tensor): codebook input of shape [B, K, T]

        Returns:
            torch.Tensor: audio output of shape [B, 1, T]

        B: batch size
        K: number of codebooks
        T: sequence length
        """

        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress the input tensor.

        Args:
            x (torch.Tensor): audio input of shape [B, 1, T]

        Returns:
            torch.Tensor: codebook output of shape [B, K, T]

        B: batch size
        K: number of codebooks
        T: sequence length
        """

        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decompress the input tensor.

        Args:
            z (torch.Tensor): codebook input of shape [B, K, T]

        Returns:
            torch.Tensor: audio output of shape [1, 1, T]

        B: batch size
        K: number of codebooks
        T: sequence length
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the audio.

        Returns:
            int: the sample rate of the audio
        """

        raise NotImplementedError
