import os
import librosa
import soundfile as sf

def resample_audio_file(input_path, output_path, target_sr=16000):
    """
    Resamples an audio file to the target sample rate and saves it.

    Parameters:
    - input_path: str, path to the input audio file
    - output_path: str, path to save the resampled audio file
    - target_sr: int, target sample rate (default 16000)
    """
    try:
        # Load the audio file with original sample rate
        audio, original_sr = librosa.load(input_path, sr=None)  # sr=None preserves original SR

        # Check if the original sample rate is different from target
        if original_sr != target_sr:
            # Resample the audio to target_sr
            audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            print(f"Resampled '{input_path}' from {original_sr} Hz to {target_sr} Hz.")
        else:
            audio_resampled = audio  # No resampling needed
            print(f"No resampling needed for '{input_path}' (already {target_sr} Hz).")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the resampled audio to the output path
        sf.write(output_path, audio_resampled, target_sr)
        print(f"Saved resampled audio to '{output_path}'.\n")

    except Exception as e:
        print(f"Error processing '{input_path}': {e}\n")

def resample_folder(input_folder, output_folder, target_sr=16000, file_extensions=None):
    """
    Resamples all audio files in a folder to the target sample rate.

    Parameters:
    - input_folder: str, path to the input folder containing audio files
    - output_folder: str, path to the folder to save resampled audio files
    - target_sr: int, target sample rate (default 16000)
    - file_extensions: list of str, audio file extensions to process
    """
    if file_extensions is None:
        # Define default audio file extensions
        file_extensions = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aiff', '.aif']

    # Convert extensions to lowercase for case-insensitive matching
    file_extensions = [ext.lower() for ext in file_extensions]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Counter for processed files
    processed_files = 0
    total_files = 0

    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            total_files += 1
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in file_extensions):
                input_path = os.path.join(root, file)

                # Compute the relative path to maintain directory structure
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Define the output file name
                base, ext = os.path.splitext(file)
                output_file = f"{base}_16k.wav"  # Saving all as .wav; modify if needed
                output_path = os.path.join(output_dir, output_file)

                # Resample and save the audio file
                resample_audio_file(input_path, output_path, target_sr=target_sr)
                processed_files += 1

    print(f"Processed {processed_files} out of {total_files} files.")

if __name__ == "__main__":
    # Define the input and output folders
    input_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/voice_recordings/'    # Replace with your input folder path
    output_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/voice_recordings_resample'  # Replace with your desired output folder path

    # Define the target sample rate
    target_sample_rate = 16000

    # Optional: Define the audio file extensions you want to process
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aiff', '.aif']

    # Start resampling
    resample_folder(input_folder, output_folder, target_sr=target_sample_rate, file_extensions=audio_extensions)
