""" 
Code for inference with the model.
"""

import audio_autoencoder
import text_encoder
import torch
from audiotools import AudioSignal
from config import Config

from model import WaveAILightning
import time


class WaveModelInference:
    """
    Class to perform inference with the model.
    """

    def __init__(self, path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = Config()

        self.model = WaveAILightning()
        if path is not None:
            self.model = WaveAILightning.load_from_checkpoint(path)

        self.model = self.model.model.to(self.device)
        self.model.eval()

        self.text_encoder = text_encoder.T5EncoderBaseModel(max_length=512)

        audio_codec_path = audio_autoencoder.utils.download(model_type="44khz")
        self.audio_codec = audio_autoencoder.DAC.load(audio_codec_path)

    def greedy_decoding(self, src_text: str):
        """Perform greedy decoding with the model.

        Args:
            src_text (str): the source text to generate the audio from
        """

        encoded_text = self.text_encoder([src_text]).to(self.device)

        input_ids = torch.zeros(1, self.config.num_codebooks, 1)
        input_ids = input_ids + self.config.pad_token_id

        # delay pattern used by Musicgen
        input_ids, mask = self.model.build_delay_pattern_mask(
            input_ids,
            pad_token_id=self.config.pad_token_id,
            max_length=self.config.max_seq_length,
        )

        input_ids = input_ids.to(self.device)
        all_tokens = [input_ids]

        for i in range(self.config.max_seq_length):
            inputs = torch.cat(all_tokens, dim=-1)
            print(inputs)
            logits = self.model(inputs, encoded_text)

            max_prob_idx = logits.argmax(dim=-1)

            output_ids = max_prob_idx[..., -1].unsqueeze(-1)

            all_tokens.append(output_ids)

            print(f"Step {i + 1} / {self.config.max_seq_length}", end="\r")

            # time.sleep(1)

        output_ids = torch.cat(all_tokens, dim=-1)
        output_ids = self.model.apply_delay_pattern_mask(output_ids.cpu(), mask)

        output_ids = output_ids[output_ids != self.config.pad_token_id].reshape(
            1, self.config.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...].squeeze(0)

        z = self.audio_codec.quantizer.from_codes(output_ids.cpu())[0]
        y = torch.tensor([])

        for i in range(0, z.shape[2], 200):
            print(f"Decoding {i} / {z.shape[2]}", end="\r")
            z_bis = z[:, :, i : i + 200]

            y_bis = self.audio_codec.decode(z_bis)
            y = torch.cat((y, y_bis), dim=-1)

        y = AudioSignal(y.detach().numpy(), sample_rate=44100)
        y.write("output.wav")


model = WaveModelInference(
    "lightning_logs/version_265/checkpoints/epoch=2-step=300.ckpt"
)

text = """ 
Heavy metal vocals with distorted electric guitar, bass guitar, and aggressive acoustic drumming.\n\nLyric: I FRILE UPOL,\nAME\nIT\nдва\nI follow you up still\nMy heart beats faster\nWhy don't copy up this chaos\nI feel so much better\nI'm blacked maniac\nI always come back\nSo I would drown my soul\nSo I will lose my soul\nFing\nStrang\nStrangy me\nFurn thing\nStratie me\nStrangy me\nYeah\nOnce you drag me down,\nMake me naked on the ground\nMeep me naked on the crown\nI've some deceived as well\nTo insults you too\nAnd to type be on my back\nbreaking for your attack\nWe hope you chips\nMissus, I'm you're so bright\nSomething\nStrangip\nStrangip\nMe\nSomething\nStrangy\nStrang in me\nI'm\nbusy\nday by day\nagain\nI'm not to play\nwith me\nso low\nspent\ntime\nyou're so\na test\nwho\nLive!\nPlug.\nOh\nYeah\nYeah\nShake on my face\nYeah, I can't break\nDon't care by my hands\nIt's real powerless\nI love you so much\nI need a kind of text\nNo I can be so near\nShut you right here\nFought me\nSturgey\nFafing\nFurn thing\nStrange you burn\nFault faith\nSunfit\nSearching me\nDDR\nMaking\nYou know,\n", "Acoustic guitar covers, guitar arpeggios, meditation/yoga background music, mellow/soft/emotional/passionate vocals, relaxing atmosphere.\n\nLyric: Billy watches the clocks.\nHis work never stops.\nHe wanders all day from cafe to cafe.\nThe love he's waiting for, she never will appear.\nShe left him standing here, but still feels near.\nHe carries a photograph that's falling apart.\nFrom being to a photograph that's falling apart.\nFrom being to a photograph.\nTouched too much by a broken heart.\nI'm going to be.\nI'm going to be.\nI'm going to\nI'm going to\nAnd there's the night\nAnd as the night\nAnd as the night\nAnd as the night falls.\nHe frequents the horns\nAnd the CD bars\nWith his beat up guitar\nWith his beat-up guitar\nHe plays some sad songs\nAnd his friends by him a beer\nNow he's glad to be here\nAway\nAway\nAnd as he heads for home through the midnight throne\nHe knows he lives in a world\nHe knows he lives in a world\nWhere he doesn't belong\nAnd that journey is end\nMy dear old\nTurn out the light\nSleep soundly tonight\nSleep soundly tonight\nSleep soundly tonight\nYou know\nYou\nYou\nYou\nYou\nYou\nI\n
""".strip()

model.greedy_decoding(text)
