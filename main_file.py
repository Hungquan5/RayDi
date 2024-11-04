import numpy as np
import torch
import librosa
import warnings
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from torch.nn import functional as F
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
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.size(0) == 0:
            raise ValueError("Empty batch received")
            
        mask = (x.sum(dim=-1) != 0).long()
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)
        
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.long(), batch_first=True)
        rnn_out, _ = self.rnn(packed_x)
        rnn_out, _ = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        
        _, unsort_idx = sort_idx.sort(0)
        rnn_out = rnn_out[unsort_idx]
        
        batch_size = rnn_out.size(0)
        last_output = rnn_out[torch.arange(batch_size), lengths[unsort_idx] - 1]
        
        x = self.layer_norm(last_output)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        
        output = self.fc2(x)
        return output

class WavFileClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.MODEL_RATE = 16000
        self.WINDOW_SIZE = self.MODEL_RATE
        self.STEP_SIZE = self.MODEL_RATE // 2
        
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512

    def load_model(self, model_path):
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
        audio, sr = librosa.load(file_path, sr=None)
        
        if sr != self.MODEL_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.MODEL_RATE)
        
        return audio

    def extract_features(self, audio_segment):
        if len(audio_segment) < self.WINDOW_SIZE:
            audio_segment = np.pad(audio_segment, 
                                 (0, self.WINDOW_SIZE - len(audio_segment)),
                                 'constant')
        elif len(audio_segment) > self.WINDOW_SIZE:
            audio_segment = audio_segment[:self.WINDOW_SIZE]
        
        mfccs = librosa.feature.mfcc(
            y=audio_segment,
            sr=self.MODEL_RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.FloatTensor(mfccs.T).unsqueeze(0)

    def process_file(self, file_path):
        """Process an audio file and return whether it contains wake words"""
        audio = self.load_and_preprocess_audio(file_path)
        
        for start in range(0, len(audio) - self.WINDOW_SIZE + 1, self.STEP_SIZE):
            segment = audio[start:start + self.WINDOW_SIZE]
            features = self.extract_features(segment)
            features = features.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                if predicted.item() == 1:
                    return True  # Return as soon as we find a wake word
                    
        return False  # No wake words found in this file

def process_wav_folder(model_path, folder_path):
    """Process all WAV files in a folder and return detection statistics"""
    classifier = WavFileClassifier(model_path)
    folder = Path(folder_path)
    
    wav_files = list(folder.glob('*.wav'))
    total_files = len(wav_files)
    files_with_detections = 0
    
    print(f"Processing {total_files} WAV files in {folder}")
    
    # Process each file
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
    model_path = 'checkpoints/model_epoch_9_4_11_run4_at1_seed46.pth'
    wav_folder = 'voice_recordings/'  # Replace with your folder path
    total_files, files_with_detections = process_wav_folder(model_path, wav_folder)

if __name__ == "__main__":
    main()