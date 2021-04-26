import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from mcl_toolbox.utils.utils import get_all_pid_for_env
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
from mcl_toolbox.analyze_sequences import analyse_sequences

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
    return df, df.iloc[0]["optimization_criterion"], df.iloc[0][
        "model"]  # get model and opti from the first row, i.e. the model and ppti with the lowest loss


def average_performance(exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True):
    """
    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num: a string "v1.0"
        pid_list: list of participant IDs
        optimization_criterion: a string
        model_index: an integer
        plotting: if true, plots are created

    Returns: average performance of model and participant

    """
    parent_directory = Path(__file__).parents[1]
    reward_info_directory = os.path.join(parent_directory, f"results/mcrl/{exp_num}/reward_{exp_num}_data")
    # create first set of data so the dataframe knows how the data is shaped

    # data_df = pd.DataFrame(columns=["Number of trials", "Reward", "Type"])
    # data_df.loc[0] = [0, 0, "algo"]

    data_temp = pickle_load(
        os.path.join(reward_info_directory, f"{pid_list[0]}_{optimization_criterion}_{model_index}.pkl"))
    number_of_trials = 35

    for pid in pid_list:
        data = pickle_load(os.path.join(reward_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"))
        data_temp["Reward"] = data_temp["Reward"].add(data["Reward"])

    # create averaged values
    data_average = data_temp
    data_average["Reward"] = data_temp["Reward"] / number_of_trials

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"results/mcrl/plots/average/")
        create_dir(plot_directory)
        plt.ylim(0, 60)
        plt.title(plot_title)
        ax = sns.lineplot(x="Number of trials", y="Reward", hue="Type", data=data_average)
        plt.savefig(f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}.png", bbox_inches='tight')
        # plt.show()
        plt.close()
    return data_average


def get_adaptive_maladaptive_participant_list(exp_num):
    """

    Args:
        exp_num: e.g. "v1.0"

    Returns: a dictionary {str: list}

    """
    _, _, _, _, _, _, _, _, _, adaptive_participants, maladaptive_participants, other_participants = analyse_sequences(
        exp_num, num_trials=35, block="training", pids=None,
        create_plot=False, number_of_top_worst_strategies=5)
    pid_dict = {"Adaptive strategies participants": adaptive_participants,
                "Maladaptive strategies participants": maladaptive_participants,
                "Other strategies participants": other_participants}
    return pid_dict

def group_by_adaptive_malapdaptive_participants(exp_num, model_index):
    """
    This function groups participants into adaptive/maladaptive/others (if they have used adaptive strategies in their last trial,
    they are adaptive) and creates plots based on the grouping

    Args:
        exp_num: e.g. "v1.0"
        model_index: the model index, e.g. 861

    Returns: nothing

    """
    pid_dict = get_adaptive_maladaptive_participant_list(exp_num)
    for plot_title, pid_list in pid_dict.items():
        average_performance(exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True)
    return None

# check whether the fitted parameters are significantly different across adaptive and maladaptive participants
def statistical_tests(exp_num):
    pid_dict = get_adaptive_maladaptive_participant_list(exp_num)

    # get the pid and find the corresponding pickle file where the parameters are stored

    # scatterplot of parameter values

if __name__ == "__main__":
    # exp_num = sys.argv[1]  # e.g. c2.1_dec
    # optimization_criterion = sys.argv[2]

    exp_num = "v1.0"
    optimization_criterion = "pseudo_likelihood"

    pid_list = get_all_pid_for_env(exp_num)

    # create a dataframe of fitted models and pid
    # df, best_optimzation_criteria, best_model = create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion)

    # calculate and plot the average performance given a model for all participants
    # average_performance(exp_num, pid_list, optimization_criterion, "Averaged performance overall", 861)

    # calculate and plot the average performance given a model after grouping participants
    group_by_adaptive_malapdaptive_participants(exp_num, model_index=861)
