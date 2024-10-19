import torch
from audiotools import AudioSignal
from transformers import AutoTokenizer

from waveai.text_encoder import T5EncoderBaseModel
from waveai.audio_autoencoder import DAC
from waveai.utils.utils import audio_format_converter, convert_to_tensor


class AudioProcessor:
    """
    Class to process audio data
    """

    def __init__(self, config):
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        self.text_enc = T5EncoderBaseModel()
        self.audio_codec = DAC()
        self.config = config

    @torch.no_grad()
    def collate_fn(self, rows: list) -> tuple:
        """Collate function for processing batches.

        Args:
            rows (list): the rows to process.

        Returns:
            tuple: the processed rows.
        """

        rows = [convert_to_tensor(row, self.config.data.audio_column) for row in rows]

        codec = [
            audio_format_converter(
                torch.permute(row[self.config.data.audio_column], (2, 1, 0)),
                self.config.model.stereo,
            )
            for row in rows
        ]
        prompt = [row[self.config.data.text_column] for row in rows]

        # Pad the codec tensors and stack them
        codes = torch.nn.utils.rnn.pad_sequence(
            codec, batch_first=True, padding_value=-100
        )

        # convert to batch x channel x num_codebooks x seq_length
        codes = codes.permute(0, 3, 2, 1).contiguous()

        # batch x (num_codebooks x channels) x seq_length
        codes = codes.view(codes.size(0), -1, codes.size(-1))

        # convert codes to long
        codes = codes.long()

        # cut the audio to the max length
        codes = codes[:, :, : self.config.model.max_seq_length]

        # encode the prompt
        prompts_embeds, prompts_masks = self.text_enc(prompt)

        # TODO: Maybe remove prompt from the return because it's a str and not a tensor (but currently used for logging)
        return codes, prompts_embeds, prompts_masks, prompt

    @torch.no_grad()
    def decode_audio(self, codec: torch.Tensor) -> AudioSignal:
        """Decode the codec tensor to audio.

        Args:
            codec (torch.Tensor): the codec tensor

        Returns:
            AudioSignal: the audio signal
        """

        y = self.audio_codec.decode(codec.cpu()).to(torch.float32)
        y = AudioSignal(y.cpu().numpy(), sample_rate=self.audio_codec.sample_rate)

        return y
