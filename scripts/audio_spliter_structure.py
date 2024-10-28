import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

def split_audio(input_folder, output_folder, segment_duration=1):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the global counter for file naming
    global_counter = 186871

    # Get all VIVOSSPK folders
    vivos_folders = [f for f in os.listdir(input_folder) if f.startswith('VIVOSSPK')]

    # Iterate through all VIVOSSPK folders with a progress bar
    for folder in tqdm(vivos_folders, desc="Processing VIVOS folders"):
        folder_path = os.path.join(input_folder, folder)
        
        # Get all WAV files in the current folder
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        # Process each WAV file in the current folder
        for filename in wav_files:
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            
            # Load the audio file
            audio = AudioSegment.from_wav(file_path)
            
            # Get audio duration in milliseconds
            duration_ms = len(audio)
            
            # Split the audio into 1-second segments
            for i in range(0, duration_ms, segment_duration * 1000):
                # Extract 1-second segment
                segment = audio[i:i + segment_duration * 1000]
                
                # Generate output filename
                output_filename = f"neg_audio_{global_counter}.wav"
                output_path = os.path.join(output_folder, output_filename)
                
                # Export the segment as WAV
                segment.export(output_path, format="wav")
                
                # Increment the global counter
                global_counter += 1

    print(f"Audio splitting completed. Total segments created: {global_counter}")

# Usage
input_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/vivos/train/waves"
output_folder = "negative_samples_audio"
split_audio(input_folder, output_folder)