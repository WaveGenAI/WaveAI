# WaveAI
A model that can make edit of music based on a prompt.

# Installation

```py
python -m pip install -r requirements.txt
```

# Download the dataset

```py
python data/download.py --audio_dir PATH
```

# Process the dataset

To process the dataset, run:
```py
python data/process.py --audio_dir PATH --output_dir PATH
```

To filter wrong description, run:
```py
python data/filter.py --audio_dir PATH --threshold 0.1 --delete
```
