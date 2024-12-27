import argparse
import os
import numpy as np
import librosa
from tqdm import tqdm
import logging
import h5py
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, data_dir, output_dir, n_mels=256, max_len=500):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_mels = n_mels
        self.max_len = max_len
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_log_mel(self, file_path):
        """Extract and normalize log-mel spectrogram features from an audio file."""
        try:
            y, sr = librosa.load(file_path, sr=16000)
            y = librosa.util.normalize(y)
            
            # Compute mel-spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=self.n_mels, window='hann'
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalize log-mel
            log_mel = (log_mel - np.mean(log_mel, axis=0)) / (np.std(log_mel, axis=0) + 1e-8)
            
            return log_mel.T  # Transpose to make time the first dimension
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def process_dataset(self, dir_0, dir_1):
        """Process the entire dataset and save features."""
        features = []
        labels = []
        filenames = []

        logger.info("Starting feature extraction...")

        # Process '0' folder with label 0
        class_path = self.data_dir / dir_0
        if class_path.exists():
            wav_files = list(class_path.glob('*.wav'))
            progress_bar = tqdm(wav_files, desc="Extracting features for class 0", total=len(wav_files))
            
            for wav_file in progress_bar:
                log_mel = self.extract_log_mel(wav_file)
                if log_mel is not None:
                    features.append(log_mel)
                    labels.append(0)
                    filenames.append(str(wav_file.name))

        # Process '1_sil' folder with label 1
        class_path = self.data_dir / dir_1
        if class_path.exists():
            wav_files = list(class_path.glob('*.wav'))
            progress_bar = tqdm(wav_files, desc="Extracting features for class 1_sil (label 1)", total=len(wav_files))
            
            for wav_file in progress_bar:
                log_mel = self.extract_log_mel(wav_file)
                if log_mel is not None:
                    features.append(log_mel)
                    labels.append(1)
                    filenames.append(str(wav_file.name))

        # Normalize across all samples
        sum_values = None
        sum_squared_values = None
        total_count = 0
        for log_mel in features:
            if sum_values is None:
                sum_values = np.sum(log_mel, axis=0, keepdims=True)
                sum_squared_values = np.sum(log_mel**2, axis=0, keepdims=True)
            else:
                sum_values += np.sum(log_mel, axis=0, keepdims=True)
                sum_squared_values += np.sum(log_mel**2, axis=0, keepdims=True)
            total_count += log_mel.shape[0]
        global_mean = sum_values / total_count
        global_variance = sum_squared_values / total_count - global_mean**2
        global_std = np.sqrt(np.maximum(global_variance, 1e-8))
        normalized_log_mels = [(log_mel - global_mean) / global_std for log_mel in features]

        # Save features and labels
        self.save_features(normalized_log_mels, labels, filenames)
        return normalized_log_mels, labels

    def save_features(self, features, labels, filenames):
        """Save extracted features using HDF5 format"""
        output_file = self.output_dir / 'processed_features.h5'
        logger.info(f"Saving features to {output_file}")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('labels', data=labels, compression='gzip')
            dt = h5py.special_dtype(vlen=str)
            filename_dset = f.create_dataset('filenames', (len(filenames),), dtype=dt)
            filename_dset[:] = filenames

            for i, feature in enumerate(features):
                f.create_dataset(f'features_{i}', data=feature, compression='gzip')

            f.attrs['n_mels'] = self.n_mels
            f.attrs['max_len'] = self.max_len
        logger.info(f"Saved {len(features)} samples")

class ProcessedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.labels = f['labels'][:]
            self.filenames = f['filenames'][:]
            self.features_list = [f[f'features_{i}'][:] for i in range(len(self.labels))]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features_list[idx], dtype=torch.float32), self.labels[idx]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Audio Feature Extraction Script")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--dir_0',type=str,required=True,help="0 label dir")
    parser.add_argument("--dir_1",type=str,required=True,help="1 label dir")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the extracted features")
    parser.add_argument('--n_mels', type=int, default=256, help="Number of MFCCs to extract (default: 13)")
    parser.add_argument('--max_len', type=int, default=500, help="Maximum length of the features (default: 500)")
    
    args = parser.parse_args()

    # Create feature extractor
    extractor = FeatureExtractor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_mels=args.n_mels,
        max_len=args.max_len
    )
    
    # Extract and save features
    features, labels = extractor.process_dataset(dir_0=args.dir_0,dir_1=args.dir_1)
    
    # Test loading the processed dataset
    logger.info("Testing processed dataset loading...")
    dataset = ProcessedAudioDataset(Path(args.output_dir) / 'processed_features.h5')
    logger.info(f"Successfully loaded {len(dataset)} samples")
    
    # Print sample information
    sample_feature, sample_label = dataset[0]
    logger.info(f"Sample feature shape: {sample_feature.shape}")
    logger.info(f"Sample label: {sample_label}")

if __name__ == "__main__":
    main()
