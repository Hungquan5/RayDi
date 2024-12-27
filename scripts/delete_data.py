import os

# Specify the folder path
folder_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_test/0_resample'

# Initialize a counter
count = 0  

# Set the number of files to delete
number_of_delete_file = 3000

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if "mixed_neg" is in the filename
    if "neg_audio" in filename:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, filename)
        
        # Try to delete the file
        try:
            os.remove(file_path)
            count += 1  # Increment the counter
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
        
        # Stop if the required number of files are deleted
        if count == number_of_delete_file:
            break

print(f"Number of files deleted with 'mixed_neg' in the name: {count}")
