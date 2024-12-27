# RayDi

## Overview
This about my custom hotword detection project using RNN. I just do it for fav and learning to enhance skills.

## Features
- Just using RNN, GRU, LSTM, easy to use.

## Installation
Follow these steps to set up the environment and dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Preprocessing
```bash
python scripts/preprocessing.py \
    --input_dir data/raw_audio \
    --output_dir data/augmented_audio \
    --noise_dir data/noise_samples \
    --target_size 10000 \
    --sample_rate 16000
```
- **`--input_dir`**: Directory containing input audio files.
- **`--output_dir`**: Directory to save augmented audio files.
- **`--noise_dir`**: Directory containing background noise audio files.
- **`--target_size`**: Target number of augmented audio files.
- **`--sample_rate`**: Sample rate for audio processing (default: 16000).
### Extracting feauture
Extract MFCC features from audio files for training and testing:
```bash
python scripts/extract_features.py \
    --data_dir data/raw_audio \
    --dir_0 0 \
    --dir_1 1_sil \
    --output_dir data/processed_features \
    --n_mfcc 13 \
    --max_len 500
```
- **`--data_dir`**: Path to the dataset directory.
- **`--dir_0`**: Subdirectory for class 0 audio files.
- **`--dir_1`**: Subdirectory for class 1 audio files.
- **`--output_dir`**: Directory to save extracted features.
- **`--n_mfcc`**: Number of MFCCs to extract (default: 13).
- **`--max_len`**: Maximum length of extracted features (default: 500).

### Training the Model
Train the RNN model using the processed features:
```bash
python scripts/train.py --input_path scripts/processed_features --epochs 20 --batch_size 32
```

### Evaluation the Model
Evaluate the trained model on a test set:
```bash
python scripts/eval.py
```
### Realtime detection with pygame
```bash
python main.py --model_path your_model.pth
```
### About other scripts 
I will add it in future. Sorry for the messy in my repo
## Performance
- **ACC**: 99%
Tested on a dataset of 10,000 audio samples with only my voice for hotword and diverse noise conditions.

## Limitations and Future Work
### Limitations
- Performance drops in highly noisy environments.
- Limited support for detecting multiple hotwords simultaneously.
- Not generalized model
### Future Work
- Implementing LSTM for better temporal context.
- Adding support for multilingual hotword detection.
- Generalized model


## Contact
For questions or suggestions, reach out at [hquan.a5@gmail.com](mailto:hquan.a5@gmail.com).
