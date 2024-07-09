# WaveAI
A model that can make edit of music based on a prompt. Follow MusicGen architecture.

![architecture](assets/image.png)

# Installation

```py
python -m pip install -r requirements.txt
```

# Download the dataset

```py
python data/download.py --download_dir PATH --mtg --nb_files_mgt 10
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

# License

See the LICENSE file.