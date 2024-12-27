import os
import shutil
import random
# Define the paths
source_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_full/0"  # Replace with the path to your source folder
destination_folder = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_test/0_resample"  # Replace with the path to your destination folder
# Number of files to copy
num_files_to_copy = 2000

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get the list of all files in the source folder
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Randomly select the desired number of files
if len(all_files) < num_files_to_copy:
    print("Not enough files in the source folder to copy!")
    num_files_to_copy = len(all_files)

random_files = random.sample(all_files, num_files_to_copy)

# Copy the files
copied_files = 0
for file_name in random_files:
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    try:
        shutil.copy2(source_file, destination_file)
        copied_files += 1
    except Exception as e:
        print(f"Error copying {file_name}: {e}")

print(f"Copied {copied_files} files to {destination_folder}")