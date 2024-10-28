import numpy as np
import torch
import librosa
import warnings
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

class ImprovedVoiceAssistantRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(ImprovedVoiceAssistantRNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Calculate lengths by finding the last non-zero index for each sequence
        mask = (x.sum(dim=-1) != 0).long()
        lengths = mask.sum(dim=1).cpu()
        
        # Ensure lengths are valid (greater than 0)
        lengths = torch.clamp(lengths, min=1)
        
        # Sort sequences by length in descending order
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        
        # Pack the sequences
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.long(), batch_first=True)
        
        # Process through RNN
        rnn_out, _ = self.rnn(packed_x)
        
        # Unpack the sequences
        rnn_out, _ = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Restore original batch order
        _, unsort_idx = sort_idx.sort(0)
        rnn_out = rnn_out[unsort_idx]
        
        # Take the last valid output for each sequence
        batch_size = rnn_out.size(0)
        last_output = rnn_out[torch.arange(batch_size), lengths[unsort_idx] - 1]
        
        x = self.batch_norm(last_output)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        output = self.fc2(x)
        return output

class WavFileClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Audio parameters
        self.MODEL_RATE = 16000  # Model's expected sample rate
        self.WINDOW_SIZE = self.MODEL_RATE  # 1 second window
        self.STEP_SIZE = self.MODEL_RATE // 2  # 0.5 second step (50% overlap)
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512

    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ImprovedVoiceAssistantRNN(
            input_size=13,
            hidden_size=256,
            output_size=2
        ).to(self.device)
        
        if "module." in list(checkpoint['model_state_dict'].keys())[0]:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file"""
        print(f"Loading audio file: {file_path}")
        
        # Load audio file with its original sample rate
        audio, sr = librosa.load(file_path, sr=None)
        print(f"Original sample rate: {sr}Hz")
        print(f"Original duration: {len(audio)/sr:.2f} seconds")
        
        # Resample if needed
        if sr != self.MODEL_RATE:
            print(f"Resampling to {self.MODEL_RATE}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.MODEL_RATE)
            print(f"Resampled duration: {len(audio)/self.MODEL_RATE:.2f} seconds")
        
        return audio

    def extract_features(self, audio_segment):
        """Extract MFCC features from an audio segment"""
        if len(audio_segment) < self.WINDOW_SIZE:
            # Pad if segment is shorter than window size
            audio_segment = np.pad(audio_segment, 
                                 (0, self.WINDOW_SIZE - len(audio_segment)),
                                 'constant')
        elif len(audio_segment) > self.WINDOW_SIZE:
            # Trim if segment is longer than window size
            audio_segment = audio_segment[:self.WINDOW_SIZE]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_segment,
            sr=self.MODEL_RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.FloatTensor(mfccs.T).unsqueeze(0)

    def process_file(self, file_path, plot=True):
        """Process an audio file and detect wake words"""
        # Load and preprocess audio
        audio = self.load_and_preprocess_audio(file_path)
        
        # Initialize results storage
        detections = []
        timestamps = []
        
        # Process audio in overlapping windows
        for start in range(0, len(audio) - self.WINDOW_SIZE + 1, self.STEP_SIZE):
            # Extract audio segment
            segment = audio[start:start + self.WINDOW_SIZE]
            
            # Extract features
            features = self.extract_features(segment)
            features = features.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                if predicted.item() == 1:
                    timestamp = start / self.MODEL_RATE
                    conf_value = confidence.item()
                    detections.append((timestamp, conf_value))
                    print(f"Wake word detected at {timestamp:.2f}s with confidence {conf_value:.2%}")
                    timestamps.append(timestamp)
        
        if plot:
            self.plot_results(audio, timestamps)
        
        return detections

    def plot_results(self, audio, timestamps):
        """Plot the audio waveform with detection markers"""
        plt.figure(figsize=(15, 5))
        
        # Plot waveform
        time = np.arange(len(audio)) / self.MODEL_RATE
        plt.plot(time, audio, 'b-', alpha=0.5, label='Audio Waveform')
        
        # Plot detection markers
        for ts in timestamps:
            plt.axvline(x=ts, color='r', alpha=0.5)
            plt.plot(ts, 0, 'ro')
        
        plt.title('Audio Waveform with Wake Word Detections')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Add legend if there are detections
        if timestamps:
            plt.plot([], [], 'r-', label='Detections')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def process_wav_folder(model_path, folder_path):
    """Process all WAV files in a folder"""
    classifier = WavFileClassifier(model_path)
    folder = Path(folder_path)
    
    # Find all WAV files in the folder
    wav_files = list(folder.glob('*.wav'))
    print(f"Found {len(wav_files)} WAV files in {folder}")
    
    # Process each file
    for wav_file in wav_files:
        print(f"\nProcessing {wav_file.name}")
        detections = classifier.process_file(str(wav_file))
        
        if not detections:
            print("No wake words detected in this file")
        print("-" * 50)

def main():
    model_path = 'checkpoints/best_model_quan.pth'
    wav_folder = 'wav_folder/'  # Replace with your folder path
    process_wav_folder(model_path, wav_folder)

if __name__ == "__main__":
    main()