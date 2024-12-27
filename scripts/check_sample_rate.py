import os
import librosa

# Path to the folder containing the audio files
folder_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_test/0_resample'

# Variable to hold the count of sample rates equal to 44100 Hz
count_44100 = 0

# Counter for processed files
file_count = 0

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.wav') or filename.endswith('.mp3'):  # Adjust extensions as needed
        file_path = os.path.join(folder_path, filename)
        
        # Load the audio file to extract the sample rate
        try:
            _, sample_rate = librosa.load(file_path, sr=None)  # sr=None ensures original sample rate is preserved
            # print(f"File: {filename}, Sample Rate: {sample_rate} Hz")
            
            # Count files with sample rate of 44100 Hz
            if sample_rate == 44100:
                count_44100 += 1
            file_count += 1
        except Exception as e:
            print(f"Could not process file {filename}: {e}")

# Print the count of files with sample rate 44100 Hz
print(f"\nProcessed {file_count} files")
print(f"Number of files with sample rate 44100 Hz: {count_44100}")
