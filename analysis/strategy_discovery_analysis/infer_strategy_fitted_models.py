import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
from vars import adaptive_pid, clicked_pid

# Directory containing the pickle files
directory = '../../results_sd_test23/mcrl/strategy_discovery_data'

# List to hold the data
data = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        # Extract the pid and model index from the filename
        pid = int(filename.split('_')[0])
        model_index = int(filename.split('_')[1])


        # Full path to the pickle file
        file_path = os.path.join(directory, filename)

        # Load the pickle file
        with open(file_path, 'rb') as f:
            file_data = pickle.load(f)
            r_values = file_data['r']

        # Append each r value along with the model index to the data list
        for r_value in r_values:
            data.append({'model_index': model_index, 'pid': pid, 'r': r_value})

# Create a DataFrame from the data list
df = pd.DataFrame(data)

def replace_values(r_list):
    return [1 if x in {13.0, 14.0, 15.0, 16.0} else 0 for x in r_list]

# filter df for adaptive_pid
df = df[df['pid'].isin(adaptive_pid)]

for model_index in df['model_index'].unique():
    model_data = df[df['model_index'] == model_index]

    ## Plot reward
    # plt.plot(list(range(1, 121)), np.mean(list(model_data['r']), axis=0), label=f'Model {model_index}')

    # replace all 13, 14, 15, 16 with 1 and all other values with 0
    model_data['optimal'] = model_data['r'].apply(replace_values)

    ## MK test
    res = mk.original_test(np.sum(list(model_data['optimal']), axis=0)/len(model_data))
    print(f"Model {model_index} MK test: {res}")

    ## Plot proportion
    plt.plot(list(range(1, 121)), np.sum(list(model_data['optimal']), axis=0)/len(model_data), label=f'Model {model_index}')

    plt.xlabel('Index')
    plt.ylabel('r')
    plt.title('r Values for Each Model')
    plt.legend()
    plt.show()
    # plt.savefig(f"plots/test23/simulated_{model_index}_clicked_pid.png")
    plt.close()