import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import os

def detect_speech(y, sr, frame_length=2048, hop_length=512, threshold_db=-40, min_silence_duration=0.1):
    """
    Detect speech segments in audio using energy-based detection with proper handling of word boundaries.
    
    Parameters:
    - y: audio time series
    - sr: sampling rate
    - frame_length: length of each frame for RMS calculation
    - hop_length: number of samples between frames
    - threshold_db: threshold in dB below reference to consider as silence
    - min_silence_duration: minimum silence duration in seconds to be considered actual silence
    
    Returns:
    - mask: boolean array indicating speech segments
    """
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True
    )[0]
    
    # Convert to dB scale
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Create initial mask based on threshold
    mask = db > threshold_db
    
    # Convert minimum silence duration to frames
    min_silence_frames = int(min_silence_duration * sr / hop_length)
    
    # Smooth the mask to avoid cutting words
    mask = smooth_mask(mask, min_silence_frames)
    
    # Expand mask to audio length
    audio_mask = np.repeat(mask, hop_length)
    
    # Ensure the mask length matches the audio length
    if len(audio_mask) > len(y):
        audio_mask = audio_mask[:len(y)]
    else:
        audio_mask = np.pad(audio_mask, (0, len(y) - len(audio_mask)))
    
    return audio_mask

def smooth_mask(mask, min_frames=50):
    """
    Smooth the mask to avoid choppy audio and preserve word boundaries.
    """
    from scipy import ndimage
    
    # Dilate first to connect nearby speech segments
    mask = ndimage.binary_dilation(mask, structure=np.ones(min_frames))
    
    # Remove small gaps
    mask = ndimage.binary_closing(mask, structure=np.ones(min_frames))
    
    # Remove noise/small segments
    mask = ndimage.binary_opening(mask, structure=np.ones(min_frames//2))
    
    return mask

def process_audio_file(input_path, output_path, threshold_db=-40, min_silence_duration=0.1):
    """
    Process a single audio file to remove silence while preserving speech continuity.
    
    Parameters:
    - input_path: path to input audio file
    - output_path: path to save processed audio file
    - threshold_db: threshold in dB below reference to consider as silence
    - min_silence_duration: minimum silence duration in seconds
    """
    print(f"Loading audio file: {input_path}")
    
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)  # sr=None preserves original sampling rate
    
    print(f"Detecting speech segments...")
    # Get speech mask
    mask = detect_speech(
        y, 
        sr, 
        threshold_db=threshold_db,
        min_silence_duration=min_silence_duration
    )
    
    # Apply mask to get speech segments
    processed_audio = y[mask]
    
    if len(processed_audio) == 0:
        print("Warning: No speech detected! Adjusting threshold...")
        # Try with a lower threshold
        mask = detect_speech(
            y, 
            sr, 
            threshold_db=threshold_db-10,
            min_silence_duration=min_silence_duration
        )
        processed_audio = y[mask]
    
    print(f"Saving processed audio to: {output_path}")
    # Save the processed audio
    sf.write(output_path, processed_audio, sr)
    
    # Print statistics
    original_duration = librosa.get_duration(y=y)
    processed_duration = librosa.get_duration(y=processed_audio)
    print(f"Original duration: {original_duration:.2f}s")
    print(f"Processed duration: {processed_duration:.2f}s")
    print(f"Removed {original_duration - processed_duration:.2f}s of silence")

def process_folder(input_folder, output_folder, threshold_db=-40, min_silence_duration=0.1):
    """
    Process all audio files in a folder.
    
    Parameters:
    - input_folder: path to folder containing audio files
    - output_folder: path to save processed audio files
    - threshold_db: threshold in dB below reference to consider as silence
    - min_silence_duration: minimum silence duration in seconds
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each audio file in the input folder
    for audio_file in Path(input_folder).glob('*.*'):
        if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
            output_path = Path(output_folder) / f"{audio_file.stem}_processed.wav"
            
            print(f"\nProcessing {audio_file.name}...")
            try:
                process_audio_file(
                    str(audio_file),
                    str(output_path),
                    threshold_db=threshold_db,
                    min_silence_duration=min_silence_duration
                )
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_full/1"
    output_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_full/1_sil"
    
    # Parameters tuned for speech
    threshold_db = -20  # in dB
    min_silence_duration = 0.07  # 100ms
    
    process_folder(input_folder, output_folder, threshold_db, min_silence_duration)