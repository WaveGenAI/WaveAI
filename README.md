# WaveAI
A model that can make edit of music based on a prompt. Follow MusicGen architecture.

# Installation

```py
python -m pip install -r requirements.txt
```

# Dataset
The dataset used is: https://huggingface.co/datasets/WaveGenAI/youtube-cc-by-music

# Launch the trainig

```py
python3 -m scripts.train --config_path config/config.yaml
```

# Run inference

```py
python3 -m scripts.inference --config_path config/config.yaml --checkpoint_path PATH
```
Then, go to http://127.0.0.1:7860

# License

See the LICENSE file.