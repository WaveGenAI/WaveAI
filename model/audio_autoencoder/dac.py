""" 
DAC Module, copied from https://huggingface.co/hance-ai/descript-audio-codec-44khz with small change to fix ram usage
"""

import dac
import numpy
import numpy as np
import torch
import torchaudio.transforms as transforms
from audiotools import AudioSignal
from transformers import PretrainedConfig, PreTrainedModel


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class DACConfig(PretrainedConfig):
    model_type = "dac"

    def __init__(
        self,
        model_type_by_sampling_freq: str = "44khz",
        encoding_chunk_size_in_sec: int = 1,
        decoding_chunk_size: int = 200,
        decoding_overlap_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        """
        Initializes the model object.
        Args:
            model_type_by_sampling_freq (str, optional): The model type based on the sampling frequency. Defaults to '44khz'. Choose among ['44khz', '24khz', '16khz']
            encoding_chunk_size_in_sec (int, optional): The size of the encoding chunk in seconds. Defaults to 1.
            decoding_chunk_size (float, optional): The decoding chunk size. Defaults to 200.
            decoding_overlap_rate (float, optional): The decoding overlap rate. Must be between 0 and 1. Defaults to 0.1.
            **kwargs: Additional keyword arguments.
        Raises:
            AssertionError: If the model_type_by_sampling_freq is not one of ['44khz', '24khz', '16khz'].
            AssertionError: If the decoding_overlap_rate is not between 0 and 1.
        """
        self.model_type_by_sampling_freq = model_type_by_sampling_freq
        self.encoding_chunk_size_in_sec = encoding_chunk_size_in_sec
        self.decoding_chunk_size = decoding_chunk_size
        self.decoding_overlap_rate = decoding_overlap_rate

        assert model_type_by_sampling_freq.lower() in ["44khz", "24khz", "16khz"]
        assert decoding_chunk_size > 1, "`decoding_chunk_size` must be greater than 1."
        assert (
            decoding_overlap_rate >= 0 and decoding_overlap_rate < 1.0
        ), "`decoding_overlap_rate` must be bewteen 0 and 1."


class DAC(PreTrainedModel):
    config_class = DACConfig

    def __init__(self, config: DACConfig = None):
        if config is None:
            config = DACConfig()

        super().__init__(config)

        self.model_type_by_sampling_freq = config.model_type_by_sampling_freq.lower()
        self.model_type_by_sampling_freq_int = {
            "44khz": 44100,
            "24khz": 24000,
            "16khz": 16000,
        }[self.model_type_by_sampling_freq]
        self.encoding_chunk_size_in_sec = config.encoding_chunk_size_in_sec
        self.decoding_chunk_size = config.decoding_chunk_size
        self.decoding_overlap_rate = config.decoding_overlap_rate

        dac_path = dac.utils.download(model_type=self.model_type_by_sampling_freq)
        self.dac = dac.DAC.load(dac_path)
        self.dac.eval()
        freeze(self.dac)

        self.downsampling_rate = int(np.prod(self.dac.encoder_rates))  # 512

    def load_audio(self, audio: str | numpy.ndarray | torch.Tensor, sr: int):
        signal = AudioSignal(audio, sample_rate=sr)
        return signal

    @torch.no_grad()
    def encode(self, audio_fname: str | numpy.ndarray | torch.Tensor, sr: int = 44100):
        self.eval()

        waveform = self.load_audio(audio_fname, sr)
        waveform = waveform.resample(self.model_type_by_sampling_freq_int)
        waveform = waveform.to_mono()

        zq, s = self._chunk_encoding(waveform, self.model_type_by_sampling_freq_int)
        return zq, s

    def _chunk_encoding(self, waveform: AudioSignal, sr: int):
        # TODO: can I make it parallel?
        """
        waveform: (c l)
        """
        x = waveform.audio_data
        chunk_size = int(self.encoding_chunk_size_in_sec * sr)

        # adjust `chunk_size` to prevent any padding in `dac.preprocess`, which causes a gap between the mini-batches in the resulting music.
        remainer = chunk_size % self.dac.hop_length
        chunk_size = chunk_size - remainer

        # process
        zq_list, s_list = [], []
        audio_length = x.shape[-1]
        for start in range(0, audio_length, chunk_size):
            end = start + chunk_size
            chunk = x[:, :, start:end]
            chunk = self.dac.preprocess(chunk, sr)
            zq, s, _, _, _ = self.dac.encode(chunk.to(self.device))
            zq = zq.cpu()
            s = s.cpu()
            """
            "zq" : Tensor[B x D x T]
                Quantized continuous representation of input
                = summation of all the residual quantized vectors across every rvq level
                = E(x) = z = \sum_n^N{zq_n} where N is the number of codebooks
            "s" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
                *first element in the N dimension = first RVQ level
            """
            zq_list.append(zq)
            s_list.append(s)
            torch.cuda.empty_cache()

        zq = torch.cat(zq_list, dim=2).float()  # (1, d, length)
        s = torch.cat(s_list, dim=2).long()  # (1, n_rvq, length)

        return zq, s

    @torch.no_grad()
    def decode(self, s: torch.IntTensor):
        """
        zq: (b, d, length)
        """
        self.eval()

        waveform = self._chunk_decoding(
            s
        )  # (b, 1, length); output always has a mono-channel.

        return waveform

    def _chunk_decoding(self, s: torch.IntTensor):
        """
        zq: (b, d, length)
        """
        length = s.shape[-1]
        chunk_size = round(int(self.decoding_chunk_size))
        overlap_size = round(
            self.decoding_overlap_rate * chunk_size
        )  # overlap size in terms of token length
        overlap_size_in_data_space = round(overlap_size * self.downsampling_rate)
        waveform_concat = None
        for start in range(0, length, chunk_size - overlap_size):
            end = start + chunk_size
            s_chunk = s[..., start:end]
            chunk = self.code_to_zq(s_chunk)
            waveform = self.dac.decode(
                chunk.to(self.device)
            )  # (b, 1, chunk_size*self.downsampling_rate)
            waveform = waveform.cpu()

            waveform_len = waveform.shape[-1]
            if waveform_len < overlap_size_in_data_space:
                overlap_size_in_data_space = waveform_len

            if isinstance(waveform_concat, type(None)):
                waveform_concat = waveform.clone()
            else:
                if self.decoding_overlap_rate != 0.0:
                    prev_x = waveform_concat[:, :, :-overlap_size_in_data_space]
                    rest_of_new_x = waveform[:, :, overlap_size_in_data_space:]
                    overlap_x_from_prev_x = waveform_concat[
                        :, :, -overlap_size_in_data_space:
                    ]  # (b, 1, overlap_size_in_data_space)
                    overlap_x_from_new_x = waveform[
                        :, :, :overlap_size_in_data_space
                    ]  # (b, 1, overlap_size_in_data_space)
                    overlap = (
                        overlap_x_from_prev_x + overlap_x_from_new_x
                    ) / 2  # take mean; maybe there's a better strategy but it seems to work fine.
                    waveform_concat = torch.cat(
                        (prev_x, overlap, rest_of_new_x), dim=-1
                    )  # (b, 1, ..)
                else:
                    prev_x = waveform_concat
                    rest_of_new_x = waveform
                    waveform_concat = torch.cat(
                        (prev_x, rest_of_new_x), dim=-1
                    )  # (b, 1, ..)
        return waveform_concat  # (b, 1, length)

    def code_to_zq(self, s: torch.IntTensor):
        """
        s: (b, n_rvq, length)
        """
        zq, _, _ = self.dac.quantizer.from_codes(
            s.to(self.device)
        )  # zq: (b, d, length)
        zq = zq.cpu()
        return zq

    def save_tensor(self, tensor: torch.Tensor, fname: str) -> None:
        torch.save(tensor.cpu(), fname)

    def load_tensor(self, fname: str):
        return torch.load(fname)

    def waveform_to_audiofile(self, waveform: torch.FloatTensor, fname: str) -> None:
        AudioSignal(waveform, sample_rate=self.model_type_by_sampling_freq_int).write(
            fname
        )
