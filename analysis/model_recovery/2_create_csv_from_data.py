import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import norm
from vars import pid_dict, hybrid_reinforce_pid_dict, mf_reinforce_pid_dict, habitual_pid_dict, non_learning_pid_dict
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.env.modified_mouselab import get_termination_mers
import sys

"""
After having fitted the model to the models again, we want to create a csv file to store the results.
Probably need to download the results from server first

This script created csv for each condition and each model. 
Each csv contains the following columns:
- pid: participant id
- model_index: model index
- condition: condition
- loss: click sequence loss
- number_of_parameters: number of parameters
- model_clicks: model clicks
- model_mer: model mer
- model_rewards: model rewards
- pid_clicks: participant clicks
- pid_mer: participant mer
- pid_rewards: participant rewards
- click_loss: click loss
- mer_loss: mer loss
"""


def click_loss(p_data, model_data, criterion, model_params=None):
    p_number_of_clicks_per_trial = [
        [len([click for click in p_clicks if click not in [0, None]]) for p_clicks in p_data]]
    a_number_of_clicks_per_trial = [
        [len([click for click in a_clicks if click not in [0, None]]) for a_clicks in algorithm_click_sequence] for
        algorithm_click_sequence in model_data]

    if criterion == "likelihood":
        sigma = 1
    else:
        if model_params:
            sigma = np.exp(model_params[0]["lik_sigma"])
        else:
            sigma = 1

    objective_value = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=sigma)
            for x, y in zip(
            a_number_of_clicks_per_trial, p_number_of_clicks_per_trial
        )
        ]
    )
    return objective_value


def mer_loss(p_mer, model_data, criterion, model_params=None):
    mean_mer = np.mean(model_data, axis=0)
    if criterion == "likelihood":
        sigma = 1
    else:
        if model_params:
            sigma = np.exp(model_params[0]["lik_sigma"])
        else:
            sigma = 1

    normal_objective = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=sigma)
            for x, y in zip(mean_mer, p_mer)
        ]
    )
    return normal_objective


def click_sequence_loss(prior_data):
    losses = [trial["result"]["loss"] for trial in prior_data[0][1]]
    return min(np.absolute(losses))


def number_of_parameters(model, criterion):
    if model in [1756, 1743]:  # no learning and habitual
        if criterion == "likelihood":
            return 3
        else:
            return 4
    elif model in [527, 522]:  # RSSL
        if criterion == "likelihood":
            return 1
        else:
            return 2
    elif model in [479, 486, 487, 490, 491, 3325, 3326]:  # reinforce and lvoc
        if criterion == "likelihood":
            return 3
        else:
            return 4
    elif model in [480, 481, 3315, 3316]:
        if criterion == "likelihood":
            return 5
        else:
            return 6
    elif model in [482, 483, 484, 485, 488, 489, 3317, 3318, 3323, 3324]:
        if criterion == "likelihood":
            return 4
        else:
            return 5
    elif model in ["level_individual", "level_level"]:
        return 8
    elif model in ["no_assumption_individual", "no_assumption_level"]:
        return 2
    elif model in ["uniform_individual", "uniform_level"]:
        return 4


def get_all_combinations(model_class, condition, pid_dict_list):
    # combines all pids with all models
    combinations = list(itertools.product([*pid_dict_list[condition]], *[model_class]))

    return combinations


def cost_function(depth):
    if depth == 0:
        return 0
    if depth == 1:
        return -1
    if depth == 2:
        return -3
    if depth == 3:
        return -30


def mapping(model_class):
    if model_class == "habitual":
        return 1743
    elif model_class == "non_learning":
        return 1756
    elif model_class == "hybrid_reinforce":
        return 3326
    elif model_class == "mf_reinforce":
        return 491
    else:
        return "model not found"


if __name__ == "__main__":
    folder_list = ["variants/3326"]
    conditions = ["v1.0", "c2.1", "c1.1",
                  "high_variance_high_cost", "high_variance_low_cost",
                  "low_variance_high_cost", "low_variance_low_cost"]
    # conditions = ["v1.0"]

    for condition in conditions:
        if condition == "strategy_discovery":
            num_trials = 120
        else:
            num_trials = 35

        for folder in folder_list:
            print("folder and condition", folder, condition)

            if folder == "hybrid_reinforce":
                pid_dict_list = hybrid_reinforce_pid_dict
            elif folder == "mf_reinforce":
                pid_dict_list = mf_reinforce_pid_dict
            elif folder == "habitual":
                pid_dict_list = habitual_pid_dict
            elif folder == "non_learning":
                pid_dict_list = non_learning_pid_dict
            elif folder.startswith("variants"):
                pid_dict_list = pid_dict
            else:
                raise ValueError("Model class not found")

            if folder.startswith("variants"):
                combinations = get_all_combinations([3315, 3316, 3317, 3318, 3323, 3324, 3325], condition,
                                                    pid_dict_list)
            else:
                combinations = get_all_combinations([491, 1743, 1756, 3326], condition, pid_dict_list)

            # create new dataframe with the columns "pid", model_index", "model_params"
            df = pd.DataFrame(columns=["pid", "model_index", "model_params"])
            # class column are the second elements of the combinations
            df["model_index"] = [x[1] for x in combinations]
            # pid column are the first elements of the combinations
            df["pid"] = [x[0] for x in combinations]

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
                    data = pd.read_pickle(
                        f'../../results_model_recovery_variants/{folder}/{condition}_data/{pid}_{model}_1.pkl')
                    model_params = pd.read_pickle(
                        f'../../results_model_recovery_variants/{folder}/{condition}_priors/{pid}_likelihood_{model}.pkl')
                except:
                    print("pid not found", pid)
                    continue

                # add loss and number of parameters
                df.at[index, "loss"] = click_sequence_loss(model_params)

                df.at[index, "number_of_parameters"] = number_of_parameters(model, criterion="likelihood")

                # add model information #todo: why str?
                df.at[index, "model_clicks"] = str(data["a"][0])
                df.at[index, "model_mer"] = str(data["mer"][0])
                df.at[index, "model_rewards"] = str(data["r"][0])

                # add participant information
                mf = ModelFitter(
                    exp_name=condition,
                    exp_attributes=exp_attributes,
                    data_path=f"../../results/",
                    number_of_trials=num_trials)

                pid_context, env = mf.get_participant_context(pid)
                pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

                df.at[index, "pid_clicks"] = str(pid_context.clicks)
                df.at[index, "pid_mer"] = str(get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline))
                df.at[index, "pid_rewards"] = str(pid_context.score)

                df.at[index, "click_loss"] = click_loss(pid_context.clicks, data["a"], criterion="likelihood",
                                                        model_params=model_params[0])
                df.at[index, "mer_loss"] = mer_loss(pid_mer, data["mer"], criterion="likelihood",
                                                    model_params=model_params[0])
                df.at[index, "model_params"] = model_params[0][0]

            # save the dataframe as csv
            df.to_csv(f"data/{folder}_{condition}.csv")
