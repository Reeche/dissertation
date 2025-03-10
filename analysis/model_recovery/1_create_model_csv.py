import numpy as np
import pandas as pd
from pathlib import Path

"""
This file to to create csv that mimic the structure of participants' mouselab-mdp data. 
So the csv can be used for fitting the models again and treat the model simulations as if they were participants.

The input are the simulation data of the models.

The resulting csv files should be uploaded to the server and treated as pid to be fitted using fit_mcrl_models.py
"""

# create csv with the columns: pid, trial_index, block, path, queries, state_rewards, score
df = pd.DataFrame(columns=["pid", "trial_index", "block", "path", "queries", "state_rewards", "score"])

# load pkl file for selected model
model = 1756 #note that tehre is no 3326 for threecond and planningamount because it was fitted before reindexing the rl_model.csv
exp = "c1.1"
dir = "non_learning"
data_path = Path(f"../../final_results/{dir}/{exp}_data")

# for each pkl in dir, check whether it contains "model" in the name
for pkl in data_path.glob("*.pkl"):
    df_temp = pd.DataFrame(columns=["pid", "trial_index", "block", "path", "queries", "state_rewards", "score"])
    if str(model) in pkl.name:
        pkl_file = pd.read_pickle(pkl)

        # for each pkl file, take the first integer before the first _ as the pid
        pid = int(pkl.name.split("_")[0])

        # populate the columns of the dataframe
        df_temp["state_rewards"] = pkl_file["envs"][0]
        df_temp["trial_index"] = list(range(0, len(pkl_file["envs"][0])))
        df_temp["pid"] = pid
        df_temp["block"] = ["training"] * len(pkl_file["envs"][0])
        df_temp["score"] = pkl_file["r"][0]

        # get path, todo: this is using the pid path not the model one because we do not save it during simulations
        pid_data = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
        pid_data = pid_data[pid_data.pid == pid]
        df_temp["path"] = pid_data["path"].values

        # get queries
        df_temp["queries"] = [{'click': {'state': {'target': list(map(str, lst))}}} for lst in pkl_file["a"][0]]

        # append to the final dataframe
        df = df._append(df_temp)

    # if not directory, create directory
    if not Path(f"../../data/model/{dir}/{exp}").exists():
        Path(f"../../data/model/{dir}/{exp}").mkdir()

    # save as csv next to "human" and data folder
    df.to_csv(f"../../data/model/{dir}/{exp}/mouselab-mdp.csv", index=False)




