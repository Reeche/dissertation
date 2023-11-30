import os.path
import itertools
import pandas as pd

"""
This file checks whether all models and PID for given experiment and setting are fitted. 
"""

pid_dict = {
    'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77,
             80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
             148, 150, 154, 155, 158, 160, 165, 169, 173],
    'c2.1': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
             79, 84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142,
             145, 149, 152, 156, 162, 164, 166, 170, 172],
    'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
             83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
             151, 153, 157, 159, 161, 163, 167, 168, 171],
    'high_variance_high_cost': [0, 1, 10, 18, 22, 25, 30, 32, 38, 41, 46, 47, 49, 57, 60, 63, 65, 70, 74, 76, 81, 83,
                                88, 89, 94, 103, 108, 109, 111, 114, 116, 118, 125, 129, 134, 139, 149, 150, 156, 161,
                                164, 167, 169, 173, 177, 182, 188, 191, 195, 198, 199, 204],
    'high_variance_low_cost': [4, 7, 8, 17, 20, 23, 26, 29, 33, 35, 40, 48, 50, 51, 53, 56, 58, 64, 71, 78, 82, 87, 92,
                               93, 95, 96, 101, 112, 117, 119, 122, 126, 131, 133, 136, 141, 145, 146, 151, 154, 158,
                               162, 168, 175, 180, 185, 187, 189, 193, 197, 202, 205],
    'low_variance_high_cost': [2, 13, 14, 16, 21, 24, 28, 31, 36, 37, 43, 45, 54, 61, 62, 68, 69, 73, 79, 80, 84, 86,
                               90, 97, 98, 100, 102, 107, 120, 124, 128, 132, 135, 138, 140, 144, 147, 153, 157, 160,
                               163, 166, 171, 174, 181, 183, 192, 194, 201, 203, 206],
    'low_variance_low_cost': [3, 5, 6, 9, 11, 12, 15, 19, 27, 34, 39, 42, 44, 52, 55, 59, 66, 67, 72, 75, 77, 85, 91,
                              99, 104, 105, 106, 110, 113, 115, 121, 123, 127, 130, 137, 142, 143, 148, 152, 155, 159,
                              165, 170, 172, 176, 178, 179, 184, 186, 190, 196, 200, 207],
    'strategy_discovery': list(range(1, 57))}

exp = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
       "low_variance_low_cost", "strategy_discovery"]

# exp_num = "low_variance_low_cost"
for exp_num in exp:

    # range of models
    # models = [522, 491, 479, 1743, 1756]
    # priors_directory = (f"results_mf_models_2000/mcrl/{exp_num}_priors")

    models = ["full"]
    priors_directory = (f"results_mb_2000_v2/mcrl/{exp_num}_mb")

    ### Create a list of all combinations and concatenate them as str with underscore
    combinations = list(itertools.product([*pid_dict[exp_num]], models))
    combination_all = []
    for combination in combinations:
        string_ints = [str(int) for int in combination]
        combination_all.append(','.join(string_ints))

    ### Get a list of all PID and model_index in the priors directory
    list_of_prior_files = os.listdir(priors_directory)
    new_list_of_prior_files = []
    for files in list_of_prior_files:
        new_list_of_prior_files.append(files.replace("likelihood_", "").replace(".pkl", "").replace("_", ","))
        # new_list_of_prior_files.append(files.replace("likelihood", "").replace(".pkl", "").replace("_", ",")) #for mb model
    # unequal items, i.e. missing PID and model_index combination
    missing_items_list = list(sorted(set(combination_all) - set(new_list_of_prior_files)))

    ### Turn the list of missing pid + model_index into 2 lists
    temp_list = []
    new_list = []
    for item in missing_items_list:
        temp_list.append(item.split(","))

    for items in temp_list:
        # empty_tuple = (int(items[0]), int(items[1])) #mf models
        empty_tuple = (int(items[0]))
        new_list.append(empty_tuple)

    # print(f"Number of missing items for {exp_num}", len(new_list)) #mf
    print(f"Number of missing items for {exp_num}:", new_list) #mb

    # save the missing pid and model_index as csv, first create a df
    # df = pd.DataFrame(new_list, columns=['pid', 'model_index'])
    # df.to_csv(f"missing_{exp_num}.csv", index=False)
