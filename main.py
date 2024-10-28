import pyaudio
import numpy as np
import torch
import librosa
import threading
import queue
import time
from scipy.io import wavfile
import warnings
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
        # Sum across feature dimension to get a 1D mask for each timestep
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
        
        # Continue with the rest of your forward logic
        x = self.batch_norm(last_output)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        output = self.fc2(x)
        return output

class RealTimeVoiceClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Audio parameters
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 1
        self.WINDOW_SIZE = self.RATE  # Display 1 second of audio
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Create queue for audio chunks
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        # Setup matplotlib for visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.fig.canvas.manager.set_window_title('Real-time Audio Visualizer')
        
        # Initialize the line plot
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.detected_text = self.ax.text(0.02, 0.95, '', 
                                        transform=self.ax.transAxes,
                                        color='red',
                                        fontsize=12,
                                        fontweight='bold')
        
        # Setup plot appearance
        self.ax.set_title('Real-time Audio Waveform', pad=10)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.WINDOW_SIZE)
        self.ax.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=0
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Buffer for continuous display
        self.audio_buffer = np.zeros(self.WINDOW_SIZE)
        
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
    
    def update_plot(self):
        """Update the audio waveform plot"""
        self.line.set_data(range(len(self.audio_buffer)), self.audio_buffer)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def extract_features(self, audio_data):
        """Extract MFCC features from audio data"""
        if len(audio_data) < self.RATE * self.RECORD_SECONDS:
            audio_data = np.pad(audio_data, 
                              (0, self.RATE * self.RECORD_SECONDS - len(audio_data)),
                              'constant')
        elif len(audio_data) > self.RATE * self.RECORD_SECONDS:
            audio_data = audio_data[:self.RATE * self.RECORD_SECONDS]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.FloatTensor(mfccs.T).unsqueeze(0)
    
    def process_audio(self):
        """Process audio chunks and make predictions"""
        buffer = np.array([], dtype=np.float32)
        detection_timer = 0
        
        while self.is_running:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                buffer = np.append(buffer, chunk)
                
                # Update the rolling display buffer
                self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
                self.audio_buffer[-len(chunk):] = chunk
                
                # Update the visualization
                self.update_plot()
                
                if len(buffer) >= self.RATE * self.RECORD_SECONDS:
                    # Extract features
                    features = self.extract_features(buffer)
                    features = features.to(self.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        if predicted.item() == 1:
                            print("Wake word detected!")
                            self.detected_text.set_text('DETECTED!')
                            detection_timer = 20  # Show detection for 20 frames
                        elif detection_timer > 0:
                            detection_timer -= 1
                            if detection_timer == 0:
                                self.detected_text.set_text('')
                    
                    # Reset buffer
                    buffer = np.array([], dtype=np.float32)
            
            time.sleep(0.01)  # Reduced sleep time for smoother animation
    
    def start(self):
        """Start the voice classification system"""
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        print("* Listening for wake word...")
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\n* Stopping...")
            self.is_running = False
            processing_thread.join()
            stream.stop_stream()
            stream.close()
            self.p.terminate()
            plt.close()

def main():
    classifier = RealTimeVoiceClassifier(
        model_path='checkpoints/best_model.pth'
    )
    classifier.start()

if __name__ == "__main__":
    main()