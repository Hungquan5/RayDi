import os

# Specify the folder path
folder_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_test/1'
count = 0  # Initialize a counter

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if "recording" is in the filename
    if "Recording" in filename:
        count += 1  # Increment the counter

print(f"Number of files with 'recording' in the name: {count}")
