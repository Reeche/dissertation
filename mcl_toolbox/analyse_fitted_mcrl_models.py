import os
import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import numpy as np

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

    Returns: A dataframe with the columns ["optimization_criterion", "model", "loss"] and index is the pid

    """
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(parent_directory, f"results/mcrl/{exp_num}/{exp_num}_priors")
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
    df["loss"] = df["loss"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    print(exp_num)
    print("Grouped model and loss", grouped_df)

    return df


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
    number_of_participants = len(pid_list)
    number_of_trials = 35

    for pid in pid_list:
        data = pickle_load(os.path.join(reward_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"))
        data_temp["Reward"] = data_temp["Reward"].add(data["Reward"])

    # create averaged values
    data_average = data_temp
    data_average["Reward"] = data_temp["Reward"] / number_of_participants

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"results/mcrl/plots/average/")
        create_dir(plot_directory)
        plt.ylim(0, 60)
        plt.title(plot_title)
        ax = sns.lineplot(x="Number of trials", y="Reward", hue="Type", data=data_average)
        plt.savefig(f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}_{plot_title}.png",
                    bbox_inches='tight')
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


def group_by_adaptive_malapdaptive_participants(exp_num, optimization_criterion, model_index=None, plotting=True):
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
        if plotting:
            average_performance(exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True)
        else:
            # get the best model for each participant group
            print(f"Grouped for {plot_title} participants")
            create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion)
            # print(f"The best model with lowest AIC for {exp_num}, {plot_title} is {best_model}")

    return None


# check whether the fitted parameters are significantly different across adaptive and maladaptive participants
def statistical_tests(exp_num, optimization_criterion, model_index):
    pid_dict = get_adaptive_maladaptive_participant_list(exp_num)

    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(parent_directory, f"results/mcrl/{exp_num}/{exp_num}_priors")

    # create the df using the dictionary keys as column headers, for this, the first file in the directory is loaded

    first_file = os.listdir(prior_directory)[1]
    first_file = os.path.join(parent_directory, f"results/mcrl/{exp_num}/{exp_num}_priors/{first_file}")
    parameter_names = pickle_load(first_file)  # header are the parameters

    df = pd.DataFrame(index=list(parameter_names[0][0].keys()),
                      columns=["Adaptive strategies participants", "Maladaptive strategies participants",
                               "Other strategies participants"])
    pid_dict_reversed = {}
    for k, v in pid_dict.items():
        for v_ in v:
            pid_dict_reversed[str(v_)] = k

    for root, dirs, files in os.walk(prior_directory, topdown=False):
        for name in files:  # iterate through each file
            if name.endswith(f"{optimization_criterion}_{model_index}.pkl"):
                try:
                    pid_ = int(name[0:3])
                except:
                    try:
                        pid_ = int(name[0:2])
                    except:
                        pid_ = int(name[0])
                plot_title = pid_dict_reversed.get(str(pid_))
                data = pickle_load(os.path.join(prior_directory, name))
                for parameters, parameter_values in data[0][0].items():
                    temp_value = df.loc[parameters, plot_title]
                    if type(temp_value) == float or temp_value is None:
                        df.at[parameters, plot_title] = [parameter_values]
                    else:
                        temp_value.append(parameter_values)
                        df.at[parameters, plot_title] = temp_value
    print(df)


            # for plot_title, pid_list in pid_dict.items():  # iterate through the list of participant ids
            #     temp_list = []
            #     for pid in pid_list:
            #         if name.startswith(f"{pid}_{optimization_criterion}_{model_index}") and name.endswith(".pkl"):
            #             data = pickle_load(os.path.join(prior_directory, name))  # load the pickle file
            #             for parameters, parameter_values in data[0][0].items():
            #                 temp_list.append(parameter_values)
            #                 df.loc[[parameters], [plot_title]] = temp_list

                # df[plot_title] = temp_list

    print(df)

    # summary statistic of the parameters
    # for columns in df:
    #     for index, row in columns.iterrows():
    #         parameter_mean = np.mean(row)
    #         print(f"Mean of parameter {index} is: {parameter_mean}")
    #         parameter_variance = np.variance(row)
    #         print(f"Variance of parameter {index} is: {parameter_variance}")
    #         print("Q1 quantile of arr : ", np.quantile(row, .25))
    #         print("Q3 quantile of arr : ", np.quantile(row, .75))
    #
    # # t-test of parameters of unequal variance
    # for index, row in df.iterrows():
    #     for plot_title_a in df:
    #         for plot_title_b in df:
    #             test_statistic, p = stats.ttest_ind(df[plot_title_a], df[plot_title_b], equal_var=False)
    #             print(f"Testing parameter {index} between {plot_title_a} vs {plot_title_b}: ")
    #             print(f"Test statistic: {test_statistic}")
    #             print(f"p-value: {p}")




if __name__ == "__main__":
    # exp_num = sys.argv[1]  # e.g. c2.1_dec
    # optimization_criterion = sys.argv[2]

    exp_num_list = ["v1.0", "c2.1_dec", "c1.1"]
    # exp_num_list = ["v1.0"]
    optimization_criterion = "pseudo_likelihood"

    model_list = [861, 1983, 1853, 1757, 1852]
    # model_list = [1983]
    statistical_tests(exp_num="v1.0", optimization_criterion="pseudo_likelihood", model_index=1853)
    # for exp_num in exp_num_list:
    #     pid_list = get_all_pid_for_env(exp_num)
    #
    #     statistical_tests(exp_num, optimization_criterion, model_index=1853)

    # create a dataframe of fitted models and pid
    # df = create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion)

    # print(f"The best model with lowest AIC for {exp_num} is {best_model}")

    # group best model by performance of participants (adaptive, maladaptive)
    # group_by_adaptive_malapdaptive_participants(exp_num, optimization_criterion, model_index=None, plotting=False)

    # for model_index in model_list:
    #     # calculate and plot the average performance given a model for all participants
    #     average_performance(exp_num, pid_list, optimization_criterion, model_index, "Averaged performance overall")
    #
    #     # calculate and plot the average performance given a model after grouping participants
    #     group_by_adaptive_malapdaptive_participants(exp_num, optimization_criterion, model_index, plotting=True)
