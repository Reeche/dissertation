import sqlite3
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from statistics import multimode

"""
Validation of whether subjective cost is correctly implemented
For this, look at the average subjective cost of fitted participants for each condition
"""

# exp_list = ["v1.0", "c2.1", "c1.1",
#             "high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost", "low_variance_high_cost"]
exp_list = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost", "low_variance_high_cost"]
# exp_list = ["low_variance_low_cost"]
sub_cost_dict = {}

learning_participants = {
    "high_variance_high_cost": [32, 49, 57, 60, 94, 108, 109, 111, 129, 134, 139, 149, 161, 164, 191, 195],
    "high_variance_low_cost": [7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 145,
                               146, 154, 158, 180, 189, 197],
    "low_variance_high_cost": [2, 13, 14, 24, 28, 36, 45, 61, 62, 69, 73, 79, 80, 86, 100, 107, 124, 128, 135, 138,
                               160, 166, 171, 174, 183, 201, 206],
    "low_variance_low_cost": [9, 42, 52, 85, 110, 115, 143, 165, 172],
    "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
             82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
             160, 165, 169, 173],
    "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
             99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
             170],
    "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
             100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
             168, 171]}

models = [489] #pseudo 481, not pseudo 489

def avg_subjective_cost_sqlite():
    for exp in exp_list:
        # Connect to the SQLite database (replace 'your_database.db' with your actual database file)
        conn = sqlite3.connect(
            f'../../../../../../Volumes/REG/MCRL_results/400_evaluations_all_models/database/{exp}_priors.db')
        cursor = conn.cursor()

        # Define the SQL query to retrieve rows where "model" is equal to 489
        query = "SELECT * FROM priors WHERE model = ?"
        model_value = 489

        # Execute the query and fetch the results
        cursor.execute(query, (model_value,))
        result = cursor.fetchall()

        # Close the database connection
        conn.close()


        temp_ = []
        # Process the result as needed
        for row in result:
            if int(row[0]) in learning_participants[exp]:
                temp = pickle.loads(row[2])
                temp_.append(temp['489'][0][0]['subjective_cost'])
        sub_cost_dict[exp] = temp_

    # calculate average subjective cost for each condition
    for exp, sub_cost in sub_cost_dict.items():
        print(exp, sum(sub_cost)/len(sub_cost))

def avg_subjective_cost_pkl(exp, models, learning_participants):
    sub_cost_dict = {}
    for model in models:
        # Directory containing the pickle files
        directory = f'../../results_subjective_cost_800/mcrl/{exp}_priors'
        temp_ = []
        # Loop through files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(f'_{model}.pkl'):
                file_integer = int(filename.split('_')[0])
                if file_integer in learning_participants[exp]:
                    file_path = os.path.join(directory, filename)
                    try:
                        with open(file_path, 'rb') as file:
                            content = pickle.load(file)
                            temp = content[0][0]['subjective_cost']
                            # if temp < 0:
                            #     print("Participant", file_integer, "has subjective cost", temp)
                            temp_.append(temp)
                            # Process the content as needed
                    except Exception as e:
                        print(f"Error loading content from {filename}: {str(e)}")
        sub_cost_dict[model] = temp_

    # calculate average subjective cost for each condition
    # for model, sub_cost in sub_cost_dict.items():
    #     print(exp, model, sum(sub_cost)/len(sub_cost))

    ##calculate the mode of the subjective cost
    for model, sub_cost in sub_cost_dict.items():
        # round the subcost to integer
        sub_cost = [round(x) for x in sub_cost]
        print(exp, model, multimode(sub_cost))

    # calculate the median of the subjective cost
    # for model, sub_cost in sub_cost_dict.items():
    #     print(exp, model, sorted(sub_cost)[len(sub_cost)//2])

    # create histograms
    # for model, sub_cost in sub_cost_dict.items():
    #     plt.hist(sub_cost, bins=6)
    #     plt.title(f'{exp} {model}')
    #     plt.savefig(f'plots/subjective_cost/{exp}_{model}_subjective_cost_8000_bin6.png')
    #     # plt.show()
    #     plt.close()

for exp in exp_list:
    avg_subjective_cost_pkl(exp, models, learning_participants)