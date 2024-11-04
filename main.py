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
import pygame
warnings.filterwarnings('ignore')

class ImprovedVoiceAssistantRNN(nn.Module):
    # [Previous RNN model code remains unchanged]
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

class PygameVoiceClassifier:
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
        self.WINDOW_SIZE = self.RATE
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Create queue for audio chunks
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.is_recording = False
        self.word_detected = False
        
        # Initialize Pygame
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Voice Detector")
        
        # Button properties
        self.button_rect = pygame.Rect(300, 500, 200, 50)
        self.button_color = (0, 128, 255)
        self.button_text = "Start Recording"
        self.font = pygame.font.Font(None, 36)
        
        # Audio visualization properties
        self.wave_points = []
        self.audio_buffer = np.zeros(self.WINDOW_SIZE)
        
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
    
    def draw_button(self):
        pygame.draw.rect(self.screen, self.button_color, self.button_rect)
        text = self.font.render(self.button_text, True, (255, 255, 255))
        text_rect = text.get_rect(center=self.button_rect.center)
        self.screen.blit(text, text_rect)
    
    def draw_waveform(self):
        # Draw audio waveform
        points = []
        for i, amp in enumerate(self.audio_buffer[::100]):  # Sample every 100th point
            x = i * (self.width / (len(self.audio_buffer) // 100))
            y = (amp * 100) + self.height // 2
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, points, 2)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
            self.audio_buffer[-len(audio_data):] = audio_data
        return (in_data, pyaudio.paContinue)
    
    def extract_features(self, audio_data):
        if len(audio_data) < self.RATE * self.RECORD_SECONDS:
            audio_data = np.pad(audio_data, 
                              (0, self.RATE * self.RECORD_SECONDS - len(audio_data)),
                              'constant')
        elif len(audio_data) > self.RATE * self.RECORD_SECONDS:
            audio_data = audio_data[:self.RATE * self.RECORD_SECONDS]
        
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.FloatTensor(mfccs.T).unsqueeze(0)
    
    def process_audio(self):
        buffer = np.array([], dtype=np.float32)
        
        while self.is_running:
            if self.is_recording and not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                buffer = np.append(buffer, chunk)
                
                if len(buffer) >= self.RATE * self.RECORD_SECONDS:
                    features = self.extract_features(buffer)
                    features = features.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        if predicted.item() == 1:
                            print("Wake word detected!")
                            self.word_detected = True
                    
                    buffer = np.array([], dtype=np.float32)
            
            time.sleep(0.01)
    
    def start(self):
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
        
        clock = pygame.time.Clock()
        
        try:
            while self.is_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.button_rect.collidepoint(event.pos):
                            self.is_recording = not self.is_recording
                            self.button_text = "Stop Recording" if self.is_recording else "Start Recording"
                            self.button_color = (255, 0, 0) if self.is_recording else (0, 128, 255)
                
                # Clear screen
                self.screen.fill((0, 0, 0))
                
                # Draw waveform
                self.draw_waveform()
                
                # Draw button
                self.draw_button()
                
                # Draw detection message
                if self.word_detected:
                    text = self.font.render("Hello Quan!", True, (255, 255, 0))
                    text_rect = text.get_rect(center=(self.width // 2, 100))
                    self.screen.blit(text, text_rect)
                
                pygame.display.flip()
                clock.tick(30)
        
        finally:
            self.is_running = False
            processing_thread.join()
            stream.stop_stream()
            stream.close()
            self.p.terminate()
            pygame.quit()

def main():
    classifier = PygameVoiceClassifier(
        model_path='checkpoints/model_epoch_10_2_11_run3_at1_seed45.pth'
    )
    classifier.start()

if __name__ == "__main__":
    main()