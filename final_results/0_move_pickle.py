import os
import shutil


def move_pickle_files(root_folder, folder_list):
    for foldername, subfolders, filenames in os.walk(root_folder):
        if foldername in [f"{root_folder}/{x}" for x in folder_list]:
            for filename in filenames:
                if filename.endswith('.pkl'):
                    source_path = os.path.join(foldername, filename)

                    foldername_save = foldername.split('/')[-1]
                    if '491' in filename:
                        destination_folder = os.path.join('mf', foldername_save)
                    elif '1756' in filename:
                        destination_folder = os.path.join('non_learning', foldername_save)
                    elif '1743' in filename:
                        destination_folder = os.path.join('habitual', foldername_save)
                    elif '3326' in filename:
                        destination_folder = os.path.join('hybrid', foldername_save)
                    else:
                        continue
                    # Create the destination folder if it doesn't exist
                    os.makedirs(destination_folder, exist_ok=True)

                    destination_path = os.path.join(destination_folder, filename)

                    # Move the file
                    shutil.move(source_path, destination_path)
                    # print(f'Moved {filename} to {destination_folder}')
                    # print("Source path: ", source_path, "Destination path: ", destination_path)


if __name__ == "__main__":
    root_folder = "../results_sd_test28/mcrl"  # Replace with the path to your root folder
    # folder_list = ["v1.0_data", "v1.0_priors", "c2.1_data", "c2.1_priors", "c1.1_data", "c1.1_priors",
    #                "high_variance_high_cost_data", "high_variance_high_cost_priors",
    #                "high_variance_low_cost_data", "high_variance_low_cost_priors",
    #                "low_variance_high_cost_data", "low_variance_high_cost_priors",
    #                "low_variance_low_cost_data", "low_variance_low_cost_priors"]  # Add other folders if needed
    folder_list = ["strategy_discovery_data", "strategy_discovery_priors"]
    move_pickle_files(root_folder, folder_list)
