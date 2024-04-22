import os
import itertools
from vars import pid_dict

"""
Iterate through all the result pickles and create one csv for each condition
Also create one csv for each condition for BMS for Matlab
"""

def get_all_combinations(model_class, condition):
    mapping = {"habitual": [1743], "non_learning": [1756], "hybrid": [491, 479], "ssl": [522], "pure": [491, 479], "mb": ["full"]}
    model_type = mapping[model_class]
    combinations = list(itertools.product([*pid_dict[condition]], [*model_type]))
    return combinations

def check_pickle_files(root_folder, folder_list):
    for target_folder in folder_list:
        if target_folder != "mb":
            for foldername, subfolders, filenames in os.walk(root_folder):
                if target_folder in foldername:
                    if "_priors" in foldername:
                        # print("folder", foldername)

                        model_class = target_folder
                        # print("model_class", model_class)
                        condition = foldername.split("/")[-1].replace("_priors", "")
                        # print("condition", condition)
                        combination_should = get_all_combinations(model_class, condition)
                        # print(combination_should)

                        combinations_found = []
                        for filename in filenames:
                            if filename.endswith('.pkl'):
                                pid_id = filename.split('_')[0]
                                model_id = filename.split('_')[2]
                                model_id = model_id.split('.')[0]
                                combinations_found.append((int(pid_id), int(model_id)))
                        # print(combinations_found)

                        # Find missing items
                        missing_items = set(combination_should) - set(combinations_found)

                        # Check if there are missing items and raise an alert
                        if missing_items:
                            print(f"Alert: The following items are missing for {model_class} and condition {condition}:")
                            for item in missing_items:
                                print(item)
                        else:
                            print(f"All items found! For folder for model_class {model_class} and condition {condition}")
        elif target_folder == "mb":
            for foldername, subfolders, filenames in os.walk(root_folder):
                if target_folder in foldername:
                    if "_mb" in foldername:
                        model_class = target_folder
                        # print("foldername", foldername)
                        condition = foldername.split("/")[-1].replace("_mb", "")
                        # print("condition", condition)
                        combination_should = get_all_combinations(model_class, condition)
                        # print(combination_should)

                        combinations_found = []
                        for filename in filenames:
                            if filename.endswith('.pkl'):
                                pid_id = filename.split('_')[0]
                                combinations_found.append((int(pid_id), "full"))
                        # print(combinations_found)

                        # Find missing items
                        missing_items = set(combination_should) - set(combinations_found)

                        # Check if there are missing items and raise an alert
                        if missing_items:
                            print(f"Alert: The following items are missing for {model_class} and condition {condition}:")
                            for item in missing_items:
                                print(item)
                        else:
                            print(f"All items found! For folder for model_class {model_class} and condition {condition}")

if __name__ == "__main__":
    root_folder = os.getcwd()
    folder_list = ["habitual", "hybrid", "non_learning", "pure", "ssl", "mb"]  # Add other folders if needed
    # folder_list = ["mb"]  # Add other folders if needed
    check_pickle_files(root_folder, folder_list)