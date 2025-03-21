import os.path
import itertools
import pandas as pd

"""
This file checks whether all models and PID for given experiment and setting are fitted. 
"""

hybrid_reinforce_pid_dict = {
    'v1.0': [5, 43, 82, 137, 154],
    'c2.1': [0, 13, 78, 166],
    'c1.1': [2, 14, 125, 171],
    'high_variance_high_cost': [83, 195],
    'high_variance_low_cost': [53],
    'low_variance_high_cost': [61, 79, 100, 107, 128, 132, 166, 206],
    'low_variance_low_cost': [42, 110, 172],
    'strategy_discovery': [3, 4, 6, 7, 9, 16, 17, 19, 23, 30, 34, 35, 41, 45, 53, 57, 58, 67, 71, 76,
                           78, 83, 86, 92, 106, 128, 133, 138, 139, 141, 143, 146, 155, 161, 164, 165,
                           167, 174, 175, 177, 184, 189, 194, 195, 201, 203, 206, 211, 216, 218, 219, 223,
                           228, 231, 232, 236, 238, 250, 255, 259, 260, 262, 267, 280, 281, 291, 292, 293,
                           299, 305, 310, 316, 317, 318, 320, 324, 327, 328, 341, 344, 347, 349, 350, 355,
                           356, 357, 359, 360, 361, 362, 373, 374, 375, 377] #n=94
}

mf_reinforce_pid_dict = {
    'v1.0': [15, 104, 148, 150, 158],
    'c2.1': [25, 31, 64, 96, 123, 128, 133, 136, 142],
    'c1.1': [7, 9, 27, 28, 42, 44, 48, 139, 163],
    'high_variance_high_cost': [108],
    'high_variance_low_cost': [197],
    'low_variance_high_cost': [2, 14, 36, 73, 98, 135, 138, 144, 157, 171, 181, 183],
    'low_variance_low_cost': [115, 137, 143, 165, 170],
    'strategy_discovery': [2, 8, 24, 43, 48, 49, 54, 62, 68, 73, 75, 80, 85, 91, 93, 96, 99, 102, 107, 110, 113, 116,
                           117, 120, 123, 124, 126, 131, 137, 145, 147, 149, 153, 156, 159, 166, 169, 171, 172, 178,
                           181, 183, 185, 187, 190, 199, 200, 207, 212, 213, 220, 221, 226, 229, 233, 242, 244, 246,
                           247, 252, 261, 263, 266, 274, 279, 286, 287, 294, 295, 296, 306, 319, 333, 337, 340, 365,
                           367, 369, 372, 376, 378] #n=81
}
habitual_pid_dict = {
    'v1.0': [1, 17, 29, 34, 38, 45, 62, 66, 80, 85, 90, 110, 155],
    'c2.1': [26, 84, 99, 113, 145, 152, 162],
    'c1.1': [12, 23, 83, 111, 116, 147],
    'high_variance_high_cost': [1, 76, 169, 191],
    'high_variance_low_cost': [35, 96],
    'low_variance_high_cost': [28, 37, 45, 69, 147],
    'low_variance_low_cost': [85, 91, 106, 186],
    'strategy_discovery': [1, 10, 11, 14, 20, 22, 25, 26, 27, 29, 33, 36, 37, 39, 40, 46, 50, 51, 52, 55, 59, 65, 70,
                           89, 95, 98, 101, 111, 115, 118, 119, 125, 129, 134, 135, 140, 142, 148, 151, 154, 162, 170,
                           180, 186, 192, 193, 202, 204, 205, 209, 210, 214, 215, 217, 234, 235, 237, 240, 241, 249,
                           253, 254, 257, 265, 268, 271, 276, 277, 282, 289, 300, 304, 308, 312, 313, 321, 322, 323,
                           329, 330, 331, 332, 339, 343, 348, 358, 363, 364, 370] #n=89
}
non_learning_pid_dict = {
    'v1.0': [6, 10, 18, 24, 56, 68, 69, 94, 106, 144, 146, 165, 173],
    'c2.1': [8, 16, 20, 22, 30, 39, 41, 49, 52, 53, 58, 60, 61, 67, 72, 86, 88, 93,
             95, 107, 108, 115, 122, 130, 134, 138, 149, 156, 164, 170, 172],
    'c1.1': [19, 36, 37, 50, 54, 65, 70, 71, 74, 81, 89, 92, 100, 102, 105, 109, 114, 131,
             135, 143, 151, 159, 167, 168],
    'high_variance_high_cost': [0, 32, 47, 57, 74, 81],
    'high_variance_low_cost': [17, 23, 154, 180],
    'low_variance_high_cost': [21, 31, 124, 201],
    'low_variance_low_cost': [12, 19, 27, 44, 52, 77, 104, 113, 130, 179, 184, 196, 200],
    'strategy_discovery': [18, 28, 32, 38, 56, 63, 72, 77, 82, 90, 103, 109, 122, 152, 173, 196, 239, 256, 275, 278,
                           309, 311, 315, 335, 336, 342, 346, 352, 353, 354, 371] #n=31
}

model_type_model_mapping = {
    "habitual": 1743,
    "mf": 491,
    "hybrid": 3326,
    "non_learning": 1756
}

exp_num = "strategy_discovery"
model_types = ["non_learning", "habitual", "mf", "hybrid"]

for model_type in model_types:
    priors_directory = (f"results_model_recovery_sd/{model_type}/strategy_discovery_priors")

    ## MB models
    # models = ["full"]
    # priors_directory = (f"results_mb_2000_v2/mcrl/{exp_num}_mb")

    ### Create a list of all combinations and concatenate them as str with underscore
    if model_type == 'hybrid':
        pid_dict = hybrid_reinforce_pid_dict
    elif model_type == 'mf':
        pid_dict = mf_reinforce_pid_dict
    elif model_type == 'habitual':
        pid_dict = habitual_pid_dict
    elif model_type == 'non_learning':
        pid_dict = non_learning_pid_dict

    # combinations = list(itertools.product([*pid_dict[exp_num]], models))
    combinations = list(itertools.product([*pid_dict[exp_num]], [491, 3326, 1743, 1756]))
    combination_all = []
    for combination in combinations:
        string_ints = [str(int) for int in combination]
        combination_all.append(','.join(string_ints))

    ### Get a list of all PID and model_index in the priors directory
    list_of_prior_files = os.listdir(priors_directory)
    new_list_of_prior_files = []
    for files in list_of_prior_files:
        new_list_of_prior_files.append(files.replace("likelihood_", "").replace(".pkl", "").replace("_", ",")) #for mf models
        # new_list_of_prior_files.append(files.replace("_likelihood", "").replace(".pkl", "").replace("_", ",")) #for mb model
    # unequal items, i.e. missing PID and model_index combination
    missing_items_list = list(sorted(set(combination_all) - set(new_list_of_prior_files)))

    ### Turn the list of missing pid + model_index into 2 lists
    temp_list = []
    new_list = []
    for item in missing_items_list:
        temp_list.append(item.split(","))

    for items in temp_list:
        empty_tuple = (int(items[0]), int(items[1])) #mf models
        # empty_tuple = (int(items[0])) #mb models
        new_list.append(empty_tuple)

    print(f"Number of missing items for {exp_num}", len(new_list)) #mf
    # print(f"Number of missing items for {exp_num}:", new_list) #mb

    # print(f"Missing items for {exp_num}:", new_list) #mf
    # save the missing pid and model_index as csv, first create a df
    ## MF models only
    df = pd.DataFrame(new_list, columns=['pid', 'model_index'])
    df.to_csv(f"missing_{model_type}.csv", index=False)
