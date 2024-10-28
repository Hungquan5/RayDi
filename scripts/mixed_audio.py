import os
import random
from pydub import AudioSegment
from pydub.effects import speedup
import numpy as np
import librosa
from tqdm import tqdm

# Constants for validation
BACKGROUND_MIN_DURATION_MS = 2000  # 2 seconds
NEGATIVE_SAMPLE_DURATION_MS = 1000  # 1 second

def apply_pitch_shift(audio, semitones):
    y = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    y_shifted = librosa.effects.pitch_shift(y.astype(float), sr=sr, n_steps=semitones)
    return AudioSegment(y_shifted.astype(np.int16).tobytes(), 
                        frame_rate=sr,
                        sample_width=audio.sample_width, 
                        channels=audio.channels)

def apply_variations(audio):
    pitch_shift = random.uniform(-2, 2)
    audio = apply_pitch_shift(audio, pitch_shift)
    
    speed_change = random.uniform(0.8, 1.2)
    audio = speedup(audio, playback_speed=speed_change)
    
    return audio

def validate_audio_length(audio, expected_length, label):
    """Validate that the audio is at least expected_length milliseconds. If not, skip."""
    if len(audio) < expected_length:
        return False, audio
    if len(audio) > expected_length:
        audio = audio[:expected_length]
    return True, audio

def mix_audio(background_folder, negative_samples_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all background audio files
    background_files = [f for f in os.listdir(background_folder) if f.endswith('.wav')]

    # Get all negative audio samples
    neg_samples = [f for f in os.listdir(negative_samples_folder) if f.endswith('.wav')]

    for neg_sample_file in tqdm(neg_samples[152003:], desc="Processing negative samples"):
        
        # Load negative sample
        neg_sample_path = os.path.join(negative_samples_folder, neg_sample_file)
        neg_sample = AudioSegment.from_wav(neg_sample_path)
        
        # Validate negative sample length (1 second)
        is_valid, neg_sample = validate_audio_length(neg_sample, NEGATIVE_SAMPLE_DURATION_MS, "Negative sample")
        if not is_valid:
            continue

        # Apply pitch and speed variations
        neg_sample = apply_variations(neg_sample)

        # Choose a random background audio file
        bg_file = random.choice(background_files)
        background = AudioSegment.from_wav(os.path.join(background_folder, bg_file))

        # Validate background length (at least 2 seconds)
        is_valid, background = validate_audio_length(background, BACKGROUND_MIN_DURATION_MS, "Background")
        if not is_valid:
            continue

        # Choose a random start point in the background
        start = random.randint(0, len(background) - len(neg_sample))
        
        # Overlay negative sample onto the background at the chosen start point
        mixed = background.overlay(neg_sample, position=start)

        # Generate output filename
        output_filename = f"mixed_{os.path.splitext(neg_sample_file)[0]}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # Export the mixed audio
        mixed.export(output_path, format="wav")

    print(f"Audio mixing completed. {len(neg_samples)} mixed audio files created.")

# Usage
background_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/preprocessing/background_samples_audio"
negative_samples_folder = "negative_samples_audio"
output_folder = "mixed_audio_output"

mix_audio(background_folder, negative_samples_folder, output_folder)
