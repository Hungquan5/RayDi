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
    def __init__(self, data_dir, output_dir, n_mfcc=13, max_len=500):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_mfcc(self, file_path):
        """Extract MFCC features from an audio file"""
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc = mfcc.T  # Transpose so that time is the first dimension
            return mfcc

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def process_dataset(self):
        """Process entire dataset and save features"""
        features = []
        labels = []
        filenames = []

        logger.info("Starting feature extraction...")

        # Process '0' folder with label 0
        class_path = self.data_dir / '0'
        if class_path.exists():
            wav_files = list(class_path.glob('*.wav'))
            progress_bar = tqdm(wav_files, desc="Extracting features for class 0", total=len(wav_files))
            
            for wav_file in progress_bar:
                mfcc = self.extract_mfcc(wav_file)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(0)
                    filenames.append(str(wav_file.name))

        # Process '1_sil' folder with label 1
        class_path = self.data_dir / '1_sil'
        if class_path.exists():
            wav_files = list(class_path.glob('*.wav'))
            progress_bar = tqdm(wav_files, desc="Extracting features for class 1_sil (label 1)", total=len(wav_files))
            
            for wav_file in progress_bar:
                mfcc = self.extract_mfcc(wav_file)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(1)  # Assigning label 1 to 1_sil files
                    filenames.append(str(wav_file.name))

        # Save features and labels
        self.save_features(features, labels, filenames)

        return features, labels

    def save_features(self, features, labels, filenames):
        """Save extracted features using HDF5 format"""
        output_file = self.output_dir / 'processed_features.h5'

        logger.info(f"Saving features to {output_file}")
        with h5py.File(output_file, 'w') as f:
            # Create datasets for labels and filenames
            f.create_dataset('labels', data=labels, compression='gzip')

            # Save filenames as attributes
            dt = h5py.special_dtype(vlen=str)
            filename_dset = f.create_dataset('filenames', (len(filenames),), dtype=dt)
            filename_dset[:] = filenames

            # Save each feature as a separate dataset
            for i, feature in enumerate(features):
                f.create_dataset(f'features_{i}', data=feature, compression='gzip')

            # Store metadata
            f.attrs['n_mfcc'] = self.n_mfcc
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
    # Configuration
    data_dir = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_full'
    output_dir = 'processed_features'
    n_mfcc = 13
    max_len = 500
    
    # Create feature extractor
    extractor = FeatureExtractor(
        data_dir=data_dir,
        output_dir=output_dir,
        n_mfcc=n_mfcc,
        max_len=max_len
    )
    
    # Extract and save features
    features, labels = extractor.process_dataset()
    
    # Test loading the processed dataset
    logger.info("Testing processed dataset loading...")
    dataset = ProcessedAudioDataset(Path(output_dir) / 'processed_features.h5')
    logger.info(f"Successfully loaded {len(dataset)} samples")
    
    # Print sample information
    sample_feature, sample_label = dataset[0]
    logger.info(f"Sample feature shape: {sample_feature.shape}")
    logger.info(f"Sample label: {sample_label}")

if __name__ == "__main__":
    main()