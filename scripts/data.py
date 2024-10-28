import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.transforms as transforms

import torchaudio.transforms as transforms

class VoiceAssistantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.audio_files = []
        self.labels = []
        self.transform = transform

        # Define a MelSpectrogram transformation
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=16000,  # Adjust based on your audio sample rate
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )

        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, label)
            files = glob.glob(os.path.join(folder_path, '*.wav'))
            self.audio_files.extend(files)
            self.labels.extend([int(label)] * len(files))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load waveform and ensure it's mono (single channel)
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:  # Convert stereo to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract Mel-Spectrogram features (which should be 2D, single channel)
        mel_spectrogram = self.mel_transform(waveform)

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
            print (mel_spectrogram.shape)
            input()
        return mel_spectrogram, torch.tensor(label)
