import argparse
import numpy as np
import torch
import librosa
import warnings
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scripts.model import RNN

warnings.filterwarnings('ignore')

class WavFileClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.MODEL_RATE = 16000
        self.WINDOW_SIZE = self.MODEL_RATE
        self.STEP_SIZE = self.MODEL_RATE // 2
        
        self.n_mfcc = 13
        self.n_fft = 1024
        self.hop_length = 512

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = RNN(
            input_size=13,
            hidden_size=256,
            output_size=2
        ).to(self.device)
        
        if "module." in list(checkpoint['model_state_dict'].keys())[0]:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def load_and_preprocess_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.util.normalize(y)
        return y

    def extract_features(self, audio_segment):
        mfccs = librosa.feature.mfcc(
            y=audio_segment,
            sr=self.MODEL_RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfccs = mfccs.T
        mfccs = (mfccs - np.mean(mfccs, axis=0)) / (np.std(mfccs, axis=0) + 1e-8)
        
        return torch.FloatTensor(mfccs).unsqueeze(0)

    def process_file(self, file_path):
        """Process an entire audio file and return whether it contains wake words."""
        audio = self.load_and_preprocess_audio(file_path)
        features = self.extract_features(audio)
        features = features.to(self.device)

        with torch.no_grad():
            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            if predicted.item() == 1:
                return True  # Wake word detected

        return False  # No wake words detected


def process_wav_folder(model_path, folder_path):
    """Process all WAV files in a folder and return detection statistics."""
    classifier = WavFileClassifier(model_path)
    folder = Path(folder_path)
    
    wav_files = list(folder.glob('*.wav'))
    total_files = len(wav_files)
    files_with_detections = 0
    
    print(f"Processing {total_files} WAV files in {folder}")
    
    for wav_file in wav_files:
        print(f"Processing {wav_file.name}...")
        has_wake_word = classifier.process_file(str(wav_file))
        
        if has_wake_word:
            files_with_detections += 1
            print(f"✓ Wake word detected in {wav_file.name}")
        else:
            print(f"✗ No wake words detected in {wav_file.name}")
    
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Files containing wake words: {files_with_detections}")
    print(f"Detection rate: {(files_with_detections/total_files)*100:.1f}%")
    
    return total_files, files_with_detections


def main():
    parser = argparse.ArgumentParser(description="Process WAV files for wake word detection.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model file."
    )
    parser.add_argument(
        "--wav_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing WAV files."
    )
    args = parser.parse_args()
    
    total_files, files_with_detections = process_wav_folder(args.model_path, args.wav_folder)


if __name__ == "__main__":
    main()
