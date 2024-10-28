import os
from pydub import AudioSegment

# Specify the input folder containing .m4a files and the output folder for .wav files
input_folder = "path/to/your/m4a/folder"
output_folder = "path/to/your/output/folder"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".m4a"):
        # Define input and output file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
        
        # Load the audio file, convert sample rate, and export as .wav
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16,000 Hz
        audio.export(output_path, format="wav")
        
        print(f"Converted: {filename} to {os.path.basename(output_path)} with 16,000 Hz sample rate")

print("Batch conversion with downsampling completed!")
