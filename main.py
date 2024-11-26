import argparse
import pyaudio
import numpy as np
import torch
import librosa
import threading
import queue
import time
import pygame
from scripts.model_RNN import RNN
from scripts.model_GRU import GRU
class PygameVoiceClassifier:
    def __init__(self, model_path, device, rate, chunk, record_seconds, n_mfcc, n_fft, hop_length):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Audio parameters
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = rate
        self.RECORD_SECONDS = record_seconds
        self.WINDOW_SIZE = self.RATE
        
        # Feature extraction parameters
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
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
        model = RNN(
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
        audio_data = librosa.util.normalize(audio_data)
        audio_data = librosa.resample(audio_data, orig_sr=self.RATE, target_sr=16000)
        
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.RATE,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfccs = mfccs.T
        # Z-score normalization
        mfccs = (mfccs - np.mean(mfccs, axis=0)) / (np.std(mfccs, axis=0) + 1e-8)
        
        return torch.FloatTensor(mfccs).unsqueeze(0)
    
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
    parser = argparse.ArgumentParser(description="Voice Classifier with PyGame and PyAudio")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the model on (cpu or cuda)")
    parser.add_argument('--rate', type=int, default=44100, help="Sampling rate for audio")
    parser.add_argument('--chunk', type=int, default=1024, help="Audio chunk size")
    parser.add_argument('--record_seconds', type=float, default=0.5, help="Duration of audio recording for classification")
    parser.add_argument('--n_mfcc', type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument('--n_fft', type=int, default=1024, help="FFT size for MFCC computation")
    parser.add_argument('--hop_length', type=int, default=512, help="Hop length for MFCC computation")
    
    args = parser.parse_args()
    
    classifier = PygameVoiceClassifier(
        model_path=args.model_path,
        device=args.device,
        rate=args.rate,
        chunk=args.chunk,
        record_seconds=args.record_seconds,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    classifier.start()

if __name__ == "__main__":
    main()
