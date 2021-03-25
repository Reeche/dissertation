import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from mcl_toolbox.utils.utils import get_all_pid_for_env
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir

"""
This file contains analysis based on fitted mcrl models.
"""


# create a dataframe of fitted models and pid
def create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion):
    """
    This function loops through all pickles and creates a dataframe with corresponding loss of each model.

    Args:
        exp_num: experiment number, e.g. c2.1_dec
        pid_list: list of pid
        optimization_criterion: e.g. "pseudo_likelihood"

    Returns: A dataframe with the columns ["optimization_criterion", "model", "loss"] and index is the pid,
            The best optimization criterien and the best model number

    """
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(parent_directory, f"results/mcrl/{exp_num}_priors")
    df = pd.DataFrame(index=pid_list, columns=["optimization_criterion", "model", "loss"])

    for root, dirs, files in os.walk(prior_directory, topdown=False):
        for name in files:
            for pid in pid_list:
                if name.startswith(f"{pid}_{optimization_criterion}") and name.endswith(".pkl"):
                    data = pickle_load(os.path.join(prior_directory, name))
                    if name[-8:-4].startswith("_"):
                        df.loc[pid]["model"] = name[-7:-4]  # get the last 3 characters, which are the model name
                    else:
                        df.loc[pid]["model"] = name[-8:-4]  # get the last 4 characters, which are the model name
                    df.loc[pid]["optimization_criterion"] = optimization_criterion
                    losses = [trial['result']['loss'] for trial in data[0][1]]
                    df.loc[pid]["loss"] = losses[0]
                    # df[pid]["AIC"] = losses[0] + 2* #todo: get the number of parameters
    df = df.sort_values(by=['loss'])
    return df, df.iloc[0]["optimization_criterion"], df.iloc[0]["model"]  # get model and opti from the first row, i.e. the model and ppti with the lowest loss

# calculate and plot the average performance given a model
def average_performance(exp_num, pid_list, optimization_criterion, model_index, plotting=True):
    """
    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num:
        pid_list:
        optimization_criterion:
        model_index:
        plotting:

    Returns: average performance of model and participant

    """
    parent_directory = Path(__file__).parents[1]
    reward_info_directory = os.path.join(parent_directory, f"results/mcrl/reward_{exp_num}_data")
    data_addition = pickle_load(os.path.join(reward_info_directory, f"{0}_{optimization_criterion}_{model_index}.pkl"))
    number_of_trials = len(data_addition) / 3 #there are three row blocks, two algo (upper and lower confidence) and one participant
    for pid in pid_list:
        data = pickle_load(os.path.join(reward_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"))
        data_addition["Reward"] = data_addition["Reward"].add(data["Reward"])

    # create averaged values
    data_average = data_addition
    data_average["Reward"] = data_average["Reward"] / number_of_trials

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"results/mcrl/plots/average/")
        create_dir(plot_directory)
        ax = sns.lineplot(x="Number of trials", y="Reward", hue="algo", data=data_average)
        plt.savefig(f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}.png", bbox_inches='tight')
        #plt.show()
        plt.close()

    return data_average


if __name__ == "__main__":
    exp_num = "v1.0"
    pid_list = get_all_pid_for_env(exp_num)
    optimization_criterion = "pseudo_likelihood"
    #create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion)
    average_performance(exp_num, pid_list, optimization_criterion, 856)
