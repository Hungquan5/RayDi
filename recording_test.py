import pyaudio
import wave
import threading

# Parameters
FORMAT = pyaudio.paInt16  # Format for recording (16-bit)
CHANNELS = 2  # Number of audio channels (stereo)
RATE = 44100  # Sample rate (Hz)
CHUNK = 1024  # Buffer size
RECORD_SECONDS = 5  # Duration of recording (seconds)
OUTPUT_FILENAME = "output.wav"  # Output file name

# Function to record audio
def record_audio():
    audio = pyaudio.PyAudio()

    # Open a new stream
    stream = audio.open(format=FORMAT, 
                        channels=CHANNELS,
                        rate=RATE, 
                        input=True, 
                        frames_per_buffer=CHUNK)
    
    print("Recording started... Press Enter to stop.")
    
    frames = []

    # Record audio in chunks for the set duration
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a .wav file
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved as {OUTPUT_FILENAME}")

def wait_for_enter():
    input("Press Enter to start recording...")
    record_audio()

# Start the process
wait_for_enter()
