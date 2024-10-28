import pyaudio
import wave
from datetime import datetime
import os
import time

def create_recordings_folder():
    """
    Create a folder for recordings if it doesn't exist
    """
    folder_name = "voice_recordings"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    return folder_name

def record_audio(duration=2):
    """
    Record audio for specified duration using PyAudio
    """
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = []
    
    # Calculate how many chunks we need for the duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Done recording!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames, RATE, FORMAT, CHANNELS

def save_recording(frames, rate, audio_format, channels, folder_name):
    """
    Save the recording as a WAV file with timestamp in specified folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    filepath = os.path.join(folder_name, filename)
    
    # Save the audio file
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Saved as {filepath}")

def main():
    # Create folder for recordings
    folder_name = create_recordings_folder()
    print(f"\nRecordings will be saved in: {os.path.abspath(folder_name)}")
    print("\nPress Enter to start recording (2 seconds each), or 'q' + Enter to quit")
    
    while True:
        user_input = input()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
            
        try:
            # Record and save audio
            frames, rate, audio_format, channels = record_audio()
            save_recording(frames, rate, audio_format, channels, folder_name)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            break

if __name__ == "__main__":
    main()