import os
import shutil

# Define the source folder
source_folder = "strategy_discovery_priors"

# Create a dictionary to map strings to destination folders
destination_folders = {
    "480": "480/strategy_discovery_priors",
    "481": "481/strategy_discovery_priors",
    "482": "482/strategy_discovery_priors",
    "483": "483/strategy_discovery_priors",
    "484": "484/strategy_discovery_priors",
    "485": "485/strategy_discovery_priors",
    "486": "486/strategy_discovery_priors",
    "487": "487/strategy_discovery_priors",
    "488": "488/strategy_discovery_priors",
    "489": "489/strategy_discovery_priors",
    "490": "490/strategy_discovery_priors",
}

# Iterate through the files in the source folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # Iterate through the keys of the destination_folders dictionary
        for key in destination_folders:
            # Check if the key (string) is in the file name
            if key in file:
                # Build the source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folders[key], file)

                # Create the destination folder if it doesn't exist
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                # Move the file to the destination folder
                shutil.move(source_path, destination_path)
                print(f"Moved {file} to {destination_folders[key]}")
                # Break the loop to move to the next file
                break