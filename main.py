import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from model.audio_autoencoder import Encodec
from tqdm import trange

from model.model import WaveAI

DEVICE = "cuda"

# set seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

musicgen = WaveAI(4, 2048, 1024, 24, 16, 1024)
musicgen = musicgen.to(DEVICE)
musicgen.load_pretrained(DEVICE)
encodec = Encodec()
encodec = encodec.to(DEVICE)
encodec.load_pretrained(DEVICE)


def sample_musicgen(seconds=0.2):
    TOP_K = 150
    TEMPERATURE = 1.0
    SPECIAL_TOKEN_ID = 2048

    tokens = (torch.ones(1, 4, 1).long() * SPECIAL_TOKEN_ID).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in trange(int(50 * seconds) + 3):
                logits = musicgen.forward(tokens, torch.zeros((1, 1, 1024)))
                topk, indices = logits[:, :, -1, :].topk(TOP_K, dim=-1)
                topk = F.softmax((topk / TEMPERATURE), dim=-1)
                samples = torch.multinomial(topk.view((-1, TOP_K)), 1).view(
                    topk.shape[:-1] + (1,)
                )
                new_tokens = torch.gather(indices, dim=-1, index=samples)
                tokens = torch.cat([tokens, new_tokens], dim=2)

    tokens = torch.stack(
        [
            tokens[:, 0, 0:-3],
            tokens[:, 1, 1:-2],
            tokens[:, 2, 2:-1],
            tokens[:, 3, 3:],
        ],
        dim=1,
    )[:, :, 1:]

    return tokens


if __name__ == "__main__":
    for seed in range(1, 100):
        # set seed to SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tokens = sample_musicgen(20)

        # Convert from tokens to audio
        manual_audio = encodec.decode(tokens)
        torchaudio.save("sample" + str(seed) + ".mp3", manual_audio[0].cpu(), 32000)
