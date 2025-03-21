import numpy as np
import pandas as pd
import os
from vars import pid_dict, get_all_combinations, cost_function, number_of_parameters

"""
Get the model parameters from priors pkl files
"""

if __name__ == "__main__":
    root_folder = os.getcwd()

    model_class = 3326
    type = "hybrid"
    conditions = ["strategy_discovery"]

    for condition in conditions:
        print(condition)
        if condition == "strategy_discovery":
            num_trials = 120
        else:
            num_trials = 35

        combinations = get_all_combinations(pid_dict, model_class, condition)

        # create new dataframe with the columns "pid", "class", model_index"
        df = pd.DataFrame(columns=["pid", "class", "model_index", "model_params"])
        # class column are the second elements of the combinations
        df["model_index"] = [x[1] for x in combinations]
        # pid column are the first elements of the combinations
        df["pid"] = [x[0] for x in combinations]

        df["class"] = type
        df["condition"] = condition

        ## setup up config for mouselap
        if condition == "high_variance_high_cost" or condition == "low_variance_high_cost":
            click_cost = 5
        elif condition == "strategy_discovery":
            click_cost = cost_function
        else:
            click_cost = 1

        exp_attributes = {
            "exclude_trials": None,
            "block": "training",
            "experiment": None,
            "click_cost": click_cost
        }

        # iterate through the df rows and get pid and model
        for index, row in df.iterrows():
            pid = row["pid"]
            model = row["model_index"]
            try:
                model_params = pd.read_pickle(f'{root_folder}/{type}/{condition}_priors/{pid}_likelihood_{model}.pkl')
            except:
                print("pid not found", pid)
                continue

            df.at[index, "number_of_parameters"] = number_of_parameters(model, criterion="likelihood")
            df.at[index, "model_params"] = model_params[0][0]

        # save the dataframe as csv
        df.to_csv(f"parameters/{type}_{model_class}_{condition}_params.csv")
