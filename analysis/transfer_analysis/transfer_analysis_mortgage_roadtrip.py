import numpy as np
import pandas as pd
import ast
import copy
import random
import pymannkendall as mk
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ranksums, ttest_rel
from scipy.special import kl_div
import json

import os
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
import collections

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

rpy2.robjects.numpy2ri.activate()
stats = importr("stats")
# utils = importr('utils')
# utils.install_packages('sm', repos="https://cloud.r-project.org")
sm = importr("sm")

"""
This file contains the plots and analysis for the transfer experiment for the mortgage and roadtrip experiment
"""

random.seed(0)


def assign_strategy_types(data, experiment_type, pre_training):
    """
    Create a df that contains the PID, number of trials and assign to each one of the 6 strategies
    Last column contains the score of each trial

    Args:
        mortgage_data: the raw data from the csv

    Returns: df with the columns pid, trial, strategy, score

    """

    # filter by pre-training or not
    if pre_training:
        data = data[data["block"] == "pretraining"]
    else:
        data = data[data["block"] == "test"]

    # find number of participants
    pid = set(data["pid"])
    number_of_pid = len(pid)
    strategy_list = []

    if experiment_type == "mortgage":
        for clicks in data["queries"]:
            clicks = ast.literal_eval(clicks)

            # remove all click sequences that are longer than 3
            if len(clicks) > 3:
                clicks.pop()

            # assign strategies
            if len(clicks) == 0:
                strategy_list.append("Frugal planning")
            elif all(elem in [8, 5, 2] for elem in clicks):
                strategy_list.append("Goal setting")
            elif clicks[0] in [8, 5, 2] and len(clicks) > 1:  # first click at the last interest rates
                strategy_list.append("Backward planning")
            elif clicks[0] in [0, 3, 6] and len(clicks) > 1:  # first click at the first interest rates
                strategy_list.append("Forward planning")
            elif clicks[0] in [1, 4, 7]:  # first click in the middle
                strategy_list.append("Middle planning")
            else:
                strategy_list.append("Other planning")

    if experiment_type == "roadtrip":
        for index, row in data.iterrows():
            # check length of destination airports
            number_of_airports = len(ast.literal_eval(row["end_nodes"]))
            clicks = ast.literal_eval(row["queries"])

            # # check if first clicks (=len of destination airport) contain the destination airport nodes
            # clicks_to_be_considered = ast.literal_eval(row["queries"])[:number_of_airports]
            #
            # # check overlap between clicks to be considered and end_nodes
            # intersection = np.intersect1d(clicks_to_be_considered, ast.literal_eval(row["end_nodes"]))
            # temp_intersection_percentage = (len(intersection) / number_of_airports)

            # get the immediate cities
            map = ast.literal_eval(row["state_rewards"])
            immediate_cities = map.get("0").get("outEdge")
            airport_cities = ast.literal_eval(row["end_nodes"])

            not_airport_not_immediate_cities = immediate_cities + airport_cities

            if len(ast.literal_eval(row["queries"])) == 0:
                strategy_list.append("Frugal planning")
            elif collections.Counter(clicks) == collections.Counter(airport_cities) or set(clicks).issubset(
                    airport_cities):
                # if clicks overlap with airport cities or are less than airport cities (because they found the one with lowest value)
                strategy_list.append("Goal setting")
            elif clicks[0] in airport_cities and len(clicks) > 1:
                strategy_list.append("Backward planning")
            elif clicks[0] in immediate_cities and len(clicks) > 1:
                strategy_list.append("Forward planning")
            elif clicks[0] not in not_airport_not_immediate_cities:
                strategy_list.append("Middle planning")
            else:
                strategy_list.append("Other planning")

    # create empty df
    df_assigned = pd.DataFrame()
    if pre_training:
        number_of_test_trials = 1
    else:
        number_of_test_trials = 15
    df_assigned["pid"] = data["pid"]
    df_assigned["trial"] = list(range(0, number_of_test_trials)) * number_of_pid
    # todo: add pretraining as label in the data, otherwise all transfer trials are labelled as "test"
    df_assigned["strategy"] = strategy_list
    df_assigned["score"] = data["score"]
    return df_assigned


def create_averages_across_trials(df):
    """
    Take the df where the strategies are assigned (df with the columns pid, trial, strategy, score)
    and groups it by trial to create a df that contains the averaged proportion of the 6 strategies across trials
    Args:
        df: df with the columns pid, trial, strategy, score
        pre_training: boolean

    Returns: df with number of trials as index, 6 (or less) strategies as column headers and their proportions in each trial

    """
    # find number of participants
    pid = set(df["pid"])
    number_of_pid = len(pid)

    # count occurences of each strategy
    grouped_df = df.groupby(['trial', 'strategy']).size()  # grouped by trial_index
    grouped_df = grouped_df.to_frame(name='count').reset_index()
    # flatten pivot table into desired shape
    grouped_df = grouped_df.pivot(index='trial', columns='strategy', values='count')
    # replace na values with 0
    grouped_df = grouped_df.fillna(0)

    # replace actual count by proportion
    grouped_df_proportions = grouped_df.div(number_of_pid)

    # add score
    score_temp = df.groupby('trial').mean()
    grouped_df_proportions["score"] = score_temp["score"]
    return grouped_df, grouped_df_proportions


def create_trial_df_for_each_participant_group(df, experiment_type, pre_training=False):
    """
    Create a df with averaged number of good nodes clicked and score grouped by trial
    Args:
        mortgage_data:
        exp_num:

    Returns: 3 different df, the first one contains the averages of good choices across all pid,
    second one returns only for adaptive participants
    last one returns only for maladaptive participants

    """

    ## Preprocess data
    # filter for mortgage or roadtrip data only
    if experiment_type == "mortgage":
        data = df.loc[df["trial_type"] == "mortgage"]
    if experiment_type == "roadtrip":
        data = df.loc[df["trial_type"] == "roadtrip"]

    df_with_strategies = assign_strategy_types(data, experiment_type, pre_training)

    # create averages for all participants
    actual_count, proportions = create_averages_across_trials(df_with_strategies)
    return df_with_strategies, proportions, actual_count


def trend_tests(grouped_df):
    ### Mann Kendall test for trend
    result_clicks = mk.original_test(grouped_df["Goal setting"])
    result_score = mk.original_test(grouped_df["score"])
    print("Mann Kendall test of goal setting", result_clicks)
    print("Mann Kendall test of score", result_score)


def create_plots_between_adaptive_maladaptive(grouped_control_df, grouped_exp_df_all, grouped_exp_df_adaptive,
                                              grouped_exp_df_maladaptive,
                                              number_of_trials):
    create_dir(f"results/cm/plots/{experiment}")

    # for each strategy and score plot the control, exp_all, adaptive, maladaptive proportions
    for columns in grouped_exp_df_all:
        try:
            if columns == "score":
                score_control_std = grouped_control_df["score"].std()
                plt.fill_between(grouped_control_df.index, grouped_control_df["score"] - score_control_std,
                                 grouped_control_df["score"] + score_control_std, alpha=0.5,
                                 label='SD of participant score')
            else:
                plt.plot(grouped_control_df.index, grouped_control_df[columns], label="Control")
        except:
            print(f"{columns} missing for experimental")
            plt.plot(grouped_control_df.index, [0] * number_of_trials, label="Control")

        try:
            if columns == "score":
                score_exp_std = grouped_exp_df_all["score"].std()
                plt.fill_between(grouped_exp_df_all.index, grouped_exp_df_all["score"] - score_exp_std,
                                 grouped_control_df["score"] + score_exp_std, alpha=0.5,
                                 label='SD of participant score')
            plt.plot(grouped_exp_df_all.index, grouped_exp_df_all[columns], label="Experimental all")
        except:
            print(f"{columns} missing for experimental")
            plt.plot(grouped_exp_df_all.index, [0] * number_of_trials, label="Experimental all")

        try:
            plt.plot(grouped_exp_df_adaptive.index, grouped_exp_df_adaptive[columns], label="Experimental adaptive")
        except:
            plt.plot(grouped_exp_df_adaptive.index, [0] * number_of_trials, label="Experimental adaptive")
            print(f"{columns} missing for experimental adaptive")

        try:
            plt.plot(grouped_exp_df_maladaptive.index, grouped_exp_df_maladaptive[columns],
                     label="Experimental maladaptive")
        except:
            print(f"{columns} missing for experimental maladaptive")
            plt.plot(grouped_exp_df_maladaptive.index, [0] * number_of_trials, label="Experimental maladaptive")

        plt.title(columns)
        plt.legend(fontsize='xx-small')
        # plt.show()

        plt.savefig(
            f"results/cm/plots/{experiment}/{columns}.png",
            bbox_inches="tight",
        )
        plt.close()

    # create plot grouped by the groups, need to drop score
    grouped_control_df = grouped_control_df.drop(columns="score")
    grouped_control_df.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Control")
    plt.savefig(f"results/cm/plots/{experiment}/control.png", bbox_inches="tight", )
    plt.close()

    grouped_exp_df_all = grouped_exp_df_all.drop(columns="score")
    grouped_exp_df_all.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental")
    plt.savefig(f"results/cm/plots/{experiment}/experimental.png", bbox_inches="tight", )
    plt.close()

    grouped_exp_df_adaptive = grouped_exp_df_adaptive.drop(columns="score")
    grouped_exp_df_adaptive.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental adaptive")
    plt.savefig(f"results/cm/plots/{experiment}/experimental_adaptive.png", bbox_inches="tight", )
    plt.close()

    grouped_exp_df_maladaptive = grouped_exp_df_maladaptive.drop(columns="score")
    grouped_exp_df_maladaptive.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental maladaptive")
    plt.savefig(f"results/cm/plots/{experiment}/experimental_maladaptive.png", bbox_inches="tight", )
    plt.close()


def mapping_strategy_to_stratety_types(experiment_type, experiment):
    """
    Maps strategy to strategy clusters defined by CM and then cluster again

    Args:
        experiment_type:
        experiment:

    Returns:

    """
    # load strategies that participants used
    strategies = pickle_load(f"results/cm/inferred_strategies/{experiment_type}_{experiment}_training/strategies.pkl")

    # for pooling the data, only the first training and last trainign are needed
    for key, values in strategies.items():
        short_values = [values[0], values[-5], values[-4], values[-3], values[-2], values[-1]]
        strategies[key] = short_values

    strategies = pd.DataFrame(strategies)

    # map strategies to clusters
    strategy_to_cluster_mapping = pickle_load(f"../../mcl_toolbox/data/kl_cluster_map.pkl")
    strategies_mapped = strategies.replace(strategy_to_cluster_mapping)

    # map clusters to clusters
    cluster_mapping = {"Goal setting": [9],
                       "Backward planning": [1, 6],
                       "Forward planning": [2, 4, 8, 11, 12],
                       "Middle planning": [3],
                       "Frugal planning": [7, 10],
                       "Other planning": [5, 13]}

    # reshape the dict by switching dict and value
    cluster_mapping_temp_dict = {}
    for k, v in cluster_mapping.items():
        for item in v:
            cluster_mapping_temp_dict[item] = k

    # mapping clusters to the 5 clustered clusters
    clusters_mapped = strategies_mapped.replace(cluster_mapping_temp_dict)
    return cluster_mapping, clusters_mapped


def get_mouselab_training_strategy_types(experiment_type, experiment):
    """
    This function creates a dataframe that counts how often a strategy has been used by one PID
    Args:
        experiment_type: str (mortgage or roadtrip)
        experiment: str (training, test)

    Returns: df with pid as rows and strategy types as columns filled with the count of that strategy type

    """
    cluster_mapping, clusters_mapped = mapping_strategy_to_stratety_types(experiment_type, experiment)

    # group them
    clusters_per_pid_count = pd.DataFrame(columns=list(cluster_mapping.keys()), index=clusters_mapped.columns)
    for index, row in clusters_per_pid_count.iterrows():  # index is pid
        for columns in clusters_per_pid_count:  # strategy name
            value = clusters_mapped[index].value_counts()
            try:
                count = value[columns]
            except:
                # print(f"Strategy {clusters_count} was not used")
                count = int(0)
            clusters_per_pid_count[columns].loc[index] = int(count)

    # # turn counts into percentage
    # clusters_per_pid_percentage = pd.DataFrame(columns=list(cluster_mapping.keys()), index=clusters_mapped.columns)
    # for index, row in clusters_per_pid_count.iterrows():
    #     for columns in clusters_per_pid_count:  # strategy name
    #         sum = row.sum()
    #         count = clusters_per_pid_count[columns].loc[index]
    #         clusters_per_pid_percentage[columns].loc[index] = count / sum

    return clusters_per_pid_count


def check_for_same_strategy_before_after_transfer(filters, experiment_type, experiment, transferdata):
    """
    This function checks how many people (in percentage) used the same stratety type (e.g. goal setting) before and after transfer,
    that is the last training trial and first transfer trial
    Args:
        experiment_type: string
        experiment: string
        transferdata: df

    Returns:
        filtered_clusters_mapped: df with pid as column and trials as rows. Each cell contains the strategy type
        filtered_transferdata: df with columns [pid, trial, strategy, score]
        percentage_of_same_strategy_types: a float containing the proportion

    """
    # find the strategy type that pid has used in the last mouselab trial
    _, clusters_mapped = mapping_strategy_to_stratety_types(experiment_type, experiment)

    ### filter out the participants who did not change their strategy in the last 3 trials
    good_partipants_list_training = []
    # get the last 3 strategies
    if filters is True:
        last_n_entries = clusters_mapped.tail(3)
    else:
        last_n_entries = clusters_mapped.tail(1)
    for column in last_n_entries:
        if len(last_n_entries[column].unique()) == 1:
            good_partipants_list_training.append(column)

    # take the intersection between the training trial and transfer trial (some where filtered out for not changing strategy)
    transfer_pid_list = transferdata["pid"].unique()
    good_participants_list = list(set(good_partipants_list_training) & set(transfer_pid_list))

    ### apply filter to training and transfer data
    filtered_clusters_mapped = clusters_mapped.filter(items=good_participants_list)
    filtered_transferdata = transferdata[transferdata['pid'].isin(good_participants_list)]

    # find the strategy type that pid has used in the first transfer trial
    # create a df with the list of pid and two columns (last strategy in training, first strategy in transfer)
    pid_strategy_before_after_transfer = pd.DataFrame(
        columns=["last_strategy_before_transfer", "first_strategy_after_transfer"],
        index=filtered_clusters_mapped.columns)

    for index, row in pid_strategy_before_after_transfer.iterrows():
        # get last row from clusters_mapped, which is the strategy on the last trial for pid (column values)
        row["last_strategy_before_transfer"] = filtered_clusters_mapped.iloc[
            -1, filtered_clusters_mapped.columns.get_loc(index)]
        # filter transferdata by pid and first strategy, i.e. trial = 0
        row["first_strategy_after_transfer"] = \
            filtered_transferdata.loc[(filtered_transferdata["pid"] == index) & (filtered_transferdata["trial"] == 0)][
                "strategy"].values[0]

    # count occurence when last two columns are the same
    comparison = np.where(
        pid_strategy_before_after_transfer["last_strategy_before_transfer"] == pid_strategy_before_after_transfer[
            "first_strategy_after_transfer"], 1, 0)  # 1 is true, 0 is false
    percentage_of_same_strategy_types = np.sum(comparison) / len(comparison)
    print("Actual number of PID using the same strategy type before and after transfer: ",
          np.sum(comparison), len(comparison))
    print("Percentage of PID using the same strategy type before and after transfer: ",
          percentage_of_same_strategy_types)
    return filtered_clusters_mapped, filtered_transferdata, percentage_of_same_strategy_types, np.sum(comparison), len(
        comparison)


def filter_pid_who_changed_strategy_during_training(data, filters):
    pid_list = data["pid"].unique()

    good_participants_list = []
    bad_participants_count = 0
    for pid in pid_list:
        strategy_list = data["strategy"][data['pid'] == pid]
        if len(strategy_list.unique()) > 1:
            good_participants_list.append(pid)
        else:
            bad_participants_count = + 1

    filtered_data = data[data['pid'].isin(good_participants_list)]
    print(bad_participants_count, " participants have been removed due to stationary strategy.")
    # include all
    if filters is False:
        filtered_data = data
    return filtered_data


def create_plots_training_test(training_exp, control_df, transfer_exp, pretraining_control, pretraining_exp):
    # need proportion of goal setting for training, control, exp, pretraining control, pretraining exp
    x_range_pretraining = range(0, 1)
    if pre_training:
        x_range_exp = range(1, (training_exp.shape[0] + 1) + len(transfer_exp))
        x_range_control = range(training_exp.shape[0] + 1, training_exp.shape[0] + len(transfer_exp) +1)
    else:
        x_range_exp = range(0, (training_exp.shape[0]) + len(transfer_exp))
        x_range_control = range(training_exp.shape[0], training_exp.shape[0] + len(transfer_exp))

    def replace_strategies(data):
        data = data.replace("Goal setting", 1)
        data = data.replace("Frugal planning", 0)
        data = data.replace("Forward planning", 0)
        data = data.replace("Middle planning", 0)
        data = data.replace("Backward planning", 0)
        data = data.replace("Other planning", 0)
        return data

    # calculate the proportions
    training_replaced = replace_strategies(training_exp)
    training_proportion = training_replaced.div(len(training_replaced.columns), axis=0).sum(axis=1)
    training_proportion = training_proportion.fillna(0)

    control = list(control_df["Goal setting"])
    experimental = list(training_proportion) + list(transfer_exp["Goal setting"])

    if pre_training:
        plt.plot(x_range_pretraining, pretraining_control["Goal setting"], 'ro', label="Pretraining Control")
        plt.plot(x_range_pretraining, pretraining_exp["Goal setting"], 'go', label="Pretraining Experimental")
    plt.plot(x_range_control, control, label="Control")
    plt.plot(x_range_exp, experimental, label="Experimental")
    plt.axvline(x=training_exp.shape[0])  # todo: if pretraining add 1
    plt.title("Proportion of Goal setting strategy")
    plt.legend()
    # plt.savefig(
    #     f"results/cm/plots/{experiment}_training_transfer.png",
    #     bbox_inches="tight",
    # )
    # plt.close()
    plt.show()
    return None


def plot_strategy_distribution(training_exp, control_df, transfer_exp, pretraining, pretraining_control,
                               pretraining_exp):
    # need: first training, last training, first exp transfer, first control transfer
    # optional pretraining exp, pretraining control
    if pretraining:
        x = ["exp pre", "control pre", "exp first train.", "exp last train.", "exp first trans.",
             "control first trans."]
    else:
        x = ["exp first training", "exp last training", "exp first transfer", "control first transfer"]
    x_axis = np.arange(len(x))

    df_training = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                        "Middle planning", "Frugal planning", "Other planning"],
                               index=range(0, number_of_training_trials))

    training_exp = training_exp.T
    for trials in training_exp:  # columns
        for columns in df_training:
            df_training[columns].iloc[trials] = len(training_exp[training_exp[trials] == columns])

    df_training = df_training.div(df_training.sum(axis=1), axis=0)

    training_proportion = df_training.div(df_training.sum(axis=1), axis=0)

    # need them as as df, therefore :1 to get the first row
    training_first = training_proportion.iloc[:1]
    training_last = training_proportion.iloc[-1:]
    exp_first_transfer = transfer_exp.iloc[:1]
    control_first_transfer = control_df.iloc[:1]

    # concatenate all
    df = pd.concat([training_first, training_last, exp_first_transfer, control_first_transfer])
    if pre_training:
        df = pd.concat([pretraining_control, pretraining_exp, training_first, training_last, exp_first_transfer, control_first_transfer])


    goal_setting = df["Goal setting"]
    backward_planning = df["Backward planning"]
    forward_planning = df["Forward planning"]
    middle_planning = df["Middle planning"]
    frugal_planning = df["Frugal planning"]
    other_planning = df["Other planning"]

    width = 0.15

    plt.bar(x_axis - width * 3, goal_setting, width=0.13, align='center', label="Goal setting")
    plt.bar(x_axis - width * 2, backward_planning, width=0.13, align='center', label="Backward planning")
    plt.bar(x_axis - width, forward_planning, width=0.13, align='center', label="Forward planning")
    plt.bar(x_axis, middle_planning, width=0.13, align='center', label="Middle planning")
    plt.bar(x_axis + width, frugal_planning, width=0.13, align='center', label="Frugal planning")
    plt.bar(x_axis + width * 2, other_planning, width=0.13, align='center', label="Other planning")
    plt.axvline(0.45)
    plt.axvline(1.45)
    plt.axvline(2.45)
    if pre_training:
        plt.axvline(3.45)
        plt.axvline(4.45)

    plt.legend(fontsize=10)
    plt.xticks(x_axis, x, fontsize=8)
    plt.savefig(f"results/cm/plots/{experiment}_strategy_distribution.png", bbox_inches="tight", )
    plt.close()
    # plt.show()
    return None


def plot_cluster_proportions(training_exp, transfer_exp):
    # exp only

    # need training strategy proportions, the input are actuals
    df_training = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                        "Middle planning", "Frugal planning", "Other planning"],
                               index=range(0, 10))

    training_exp = training_exp.T
    for trials in training_exp:  # columns
        for columns in df_training:
            df_training[columns].iloc[trials] = len(training_exp[training_exp[trials] == columns])

    df_training = df_training.div(df_training.sum(axis=1), axis=0)

    df_transfer = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                        "Middle planning", "Frugal planning", "Other planning"],
                               index=range(0, 15))

    # need transfer strategy proportions, the input are actuals
    trials = range(0, 15)
    for columns in df_transfer:
        for trial in trials:
            temp_df = transfer_exp[transfer_exp["trial"] == trial]
            try:
                count = len(temp_df[temp_df["strategy"] == columns])
            except:  # if not found
                count = 0
            df_transfer[columns].iloc[trial] = count

    df_transfer = df_transfer.div(df_transfer.sum(axis=1), axis=0)

    # fuse both pd into one
    frames = [df_training, df_transfer]
    df = pd.concat(frames)

    x_range = range(0, df.shape[0])
    plt.plot(x_range, df)
    plt.axvline(x=training_exp.shape[1])
    # plt.title("Proportion of Goal setting strategy")
    plt.legend(df.columns, loc='upper right', prop={'size': 12})
    plt.savefig(
        f"results/cm/plots/{experiment}_strategy_proportions_training_transfer.png",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()

    return None


def kl1(training, transfer, exp_data):
    """
    We use a permutation test to test whether the KL[first transfer trial; first training trial] and
    KL[first transfer trial; last training trial] are significantly different. Only for experimental condition

    Args:
        training:
        transfer:
        exp_data:

    Returns:

    """
    # need strategy distribution of the first training trial, last training trial and first transfer trial
    training_first_last_trials = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                       "Middle planning", "Frugal planning", "Other planning"])

    # get proportion of first and last row of training
    for columns in training_first_last_trials:
        # count occurrence of training_df columns, i.e. the strategy types
        first_trial_proportion = list(training.iloc[1]).count(columns) / len(training.iloc[1])
        last_trial_proportion = list(training.iloc[-1]).count(columns) / len(training.iloc[-1])
        training_first_last_trials[columns] = [first_trial_proportion, last_trial_proportion]

    # drop score
    exp_data = exp_data.drop(columns="score")
    exp_data.rename(columns={"Other strategy": "Other planning"})

    # todo: hacky way to add missing column
    if experiment == "mortgage":
        exp_data["Other planning"] = 0

    # add a very small value to the 0 in exp_data and training_first_last_trials
    exp_data = exp_data.replace(0, 0.00001)
    training_first_last_trials = training_first_last_trials.replace(0, 0.00001)

    # make sure all pd columns align
    exp_data = exp_data.reindex(sorted(exp_data.columns), axis=1)
    training_first_last_trials = training_first_last_trials.reindex(sorted(training_first_last_trials.columns), axis=1)

    first_training = np.sum(kl_div(list(exp_data.iloc[0]), list(training_first_last_trials.iloc[0])))
    last_training = np.sum(kl_div(list(exp_data.iloc[0]), list(training_first_last_trials.iloc[-1])))

    gT = first_training - last_training

    # create df with columns containing the strategy used by PID for first, last training trial and first transfer trial
    training = training.T
    strategy_table = pd.DataFrame(columns=["first_training", "last_training", "first_transfer"], index=training.index)
    for index, row in strategy_table.iterrows():
        strategy_table.loc[index]["first_training"] = training.loc[index][0]
        strategy_table.loc[index]["last_training"] = training.loc[index][9]
        # strategy_table.loc[index]["first_transfer"] = transfer.loc[index][0]
        strategy_table.loc[index]["first_transfer"] = \
            transfer.loc[(transfer['pid'] == index) & (transfer["trial"] == 0)]["strategy"].item()

    # Copy pooled distribution:
    pS = copy.copy(strategy_table)
    # Initialize permutation:
    pD = []
    # Define p (number of permutations), i.e. length of pD:
    p = 10000
    # Permutation loop:
    for i in range(0, p):
        # # Shuffle the data: randomly swap cells in the columns last_training and first_training
        idx = np.random.rand(len(pS)) < 0.5
        pS.loc[idx, ['first_training', 'last_training']] = pS.loc[idx, ['last_training', 'first_training']].to_numpy()

        # calculate the average after shuffle, 0 is first training, 1 is last training, 2 is transfer
        average_after_shuffle = pd.DataFrame(columns=exp_data.columns, index=[0, 1, 2])
        # for columns in pS:
        for columns in exp_data.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["first_training"].value_counts()[columns] / len(pS.index)
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[0][columns] = proportion

        for columns in exp_data.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["last_training"].value_counts()[columns] / len(pS.index)
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[1][columns] = proportion

        for columns in exp_data.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["first_transfer"].value_counts()[columns] / len(pS.index)
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[2][columns] = proportion

        # calculate KL across the shuffled, averaged values
        average_after_shuffle = average_after_shuffle.replace(0, 0.00001)
        shuffled_kl_first = np.sum(kl_div(list(average_after_shuffle.iloc[0]), list(average_after_shuffle.iloc[2])))
        shuffled_kl_last = np.sum(kl_div(list(average_after_shuffle.iloc[1]), list(average_after_shuffle.iloc[2])))

        pD.append(shuffled_kl_first - shuffled_kl_last)

    p_val = len(np.where(pD >= gT)[0]) / (p * len(training_first_last_trials.columns))
    # p_val = len(np.where(pD >= gT)) / (p * len(training_first_last_trials.columns))
    print("p value of permutation test ", p_val)

    return None


def kl2(training_exp, transfer_exp, exp_data, control_transfer_count, control_data):
    """
    We use a permutation test to test whether the KL[first transfer trial; first training trial] and
    KL[first transfer trial; last training trial] are significantly different. Only for experimental condition

    We need: control transfer trials; experimental training trials; experimental transfer trials
    Args:
        training_exp: df with pid as columns and trials as rows. Each cell contains the strategy used
        transfer_exp: ungrouped df
        exp_data: df with strategy as columns, trials as rows. Each cell contains the frequency of the strategy
        control_transfer: df with strategy as columns, trials as rows. Each cell contains the COUNT of the strategy

    Returns:

    """
    # get exp training last trial
    exp_training_last_trial = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                    "Middle planning", "Frugal planning", "Other planning"])
    for columns in exp_training_last_trial:
        exp_training_last_trial[columns] = [list(training_exp.iloc[-1]).count(columns) / len(training_exp.iloc[-1])]

    # get control_transfer proportions
    control_transfer_proportions = control_transfer_count.div(control_transfer_count.sum(axis=1), axis=0)
    # todo: change the hacky solution to add "Other planning" manually because no participant used it
    if experiment == "roadtrip":
        control_transfer_proportions["Other planning"] = 0
    if experiment == "mortgage":
        exp_data["Other planning"] = 0
        control_transfer_proportions["Other planning"] = 0

    # drop score
    exp_data = exp_data.drop(columns="score")
    exp_data.rename(columns={"Other strategy": "Other planning"})

    # add a very small value to the 0 in exp_data and training_first_last_trials
    exp_training_last_trial = exp_training_last_trial.replace(0, 0.00001)
    exp_data = exp_data.replace(0, 0.00001)
    control_transfer_proportions = control_transfer_proportions.replace(0, 0.00001)

    # sort the columns so exp and control columns have the same order
    exp_training_last_trial = exp_training_last_trial.reindex(sorted(exp_training_last_trial.columns), axis=1)
    control_transfer_proportions = control_transfer_proportions.reindex(sorted(control_transfer_proportions.columns),
                                                                        axis=1)

    # kl[exp last training; exp first transfer]
    kl_exp_training_exp_transfer = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(exp_data.iloc[0])))
    # kl[exp last training; control first transfer]
    kl_exp_training_control_transfer = np.sum(
        kl_div(list(exp_training_last_trial.iloc[0]), list(control_transfer_proportions.iloc[0])))

    gT = kl_exp_training_exp_transfer - kl_exp_training_control_transfer

    # create dict with keys "control_transfer", "exp_transfer", "first_training"
    training_exp = training_exp.T
    strategy_dict = {}
    strategy_dict["first_training"] = training_exp.iloc[:, 0].values  # get first column
    strategy_dict["exp_transfer"] = transfer_exp.loc[(transfer_exp["trial"] == 0)]["strategy"].values
    strategy_dict["control_transfer"] = control_data.loc[(control_data["trial"] == 0)]["strategy"].values

    # Copy pooled distribution:
    pS = copy.copy(strategy_dict)
    # Initialize permutation:
    pD = []
    # Define p (number of permutations), i.e. length of pD:
    p = 10000
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data: randomly swap cells in the columns exp_transfer and control_transfer
        # pool the values from exp_transfer and control_transfer together
        pool = list(pS["control_transfer"]) + list(pS["exp_transfer"])
        random.shuffle(pool)

        # get first n items, should have same length as entries in strategy_dict["control_transfer"], i.e. 15
        pS["control_transfer"] = pool[0:len(pS["control_transfer"])]
        # get last n items, , should have same length as entries in strategy_dict["exp_transfer"], i.e. 18
        pS["exp_transfer"] = pool[-len(pS["exp_transfer"]):]

        # calculate the average after shuffle, 0 is control transfer, 1 is exp transfer
        average_after_shuffle = pd.DataFrame(columns=exp_data.columns, index=[0, 1])
        # for columns in pS:
        for columns in exp_data.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["control_transfer"].count(columns) / len(pS["control_transfer"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[0][columns] = proportion

        for columns in exp_data.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["exp_transfer"].count(columns) / len(pS["exp_transfer"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[1][columns] = proportion

        # calculate KL across the shuffled, averaged values
        average_after_shuffle = average_after_shuffle.replace(0, 0.00001)
        shuffled_kl_first = np.sum(kl_div(list(average_after_shuffle.iloc[0]), list(exp_training_last_trial.iloc[0])))
        shuffled_kl_last = np.sum(kl_div(list(average_after_shuffle.iloc[1]), list(exp_training_last_trial.iloc[0])))

        pD.append(shuffled_kl_first - shuffled_kl_last)

    p_val = len(np.where(pD <= gT)[0]) / (p * len(exp_data.columns))
    print("p value of permutation test ", p_val)

    return None


def kl3(training_exp, transfer_exp_prop, pretraining_exp, pretraining_count, transfer_count):
    """
     KL[exp transfer; exp last training] < KL[exp pretraining; exp last training]
    only experimental data
    Args:
        training_exp: df with pid and their corresponding strategy name for each trial
        transfer_exp_prop: grouped_exp_df
        pretraining_exp: grouped_exp_pretraining_df
        pretraining_count: non_grouped_exp_pretraining_df; bad PID are not removed yet
        transfer_count: transfer_exp

    Returns: p-value of the permutation test

    """
    # KL[exp transfer; exp last training] < KL[exp pretraining; exp last training]
    # only experimental data

    # get exp training last trial and proportions
    exp_training_last_trial = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                    "Middle planning", "Frugal planning", "Other planning"])
    for columns in exp_training_last_trial:
        exp_training_last_trial[columns] = [list(training_exp.iloc[-1]).count(columns) / len(training_exp.iloc[-1])]

    pretraining_exp = pretraining_exp.drop(columns=["score"])
    transfer_exp_prop = transfer_exp_prop.drop(columns=["score"])

    # miss the missing columns, i.e. strategy types for both transfer and pretraining
    if exp_training_last_trial.shape[1] != pretraining_exp.shape[1]:
        missing_columns = exp_training_last_trial.columns.difference(pretraining_exp.columns).tolist()
        for column in missing_columns:
            pretraining_exp[column] = 0

    if exp_training_last_trial.shape[1] != transfer_exp_prop.shape[1]:
        missing_columns = exp_training_last_trial.columns.difference(transfer_exp_prop.columns).tolist()
        for column in missing_columns:
            transfer_exp_prop[column] = 0

    # sort the df so now all column names align
    exp_training_last_trial = exp_training_last_trial.reindex(sorted(exp_training_last_trial.columns), axis=1)
    pretraining_exp = pretraining_exp.reindex(sorted(pretraining_exp.columns), axis=1)
    transfer_exp_prop = transfer_exp_prop.reindex(sorted(transfer_exp_prop.columns), axis=1)

    # add a very small value to the 0 in exp_data and training_first_last_trials
    exp_training_last_trial = exp_training_last_trial.replace(0, 0.00001)
    pretraining_exp = pretraining_exp.replace(0, 0.00001)
    transfer_exp_prop = transfer_exp_prop.replace(0, 0.00001)

    # KL[exp last training; exp first transfer]
    kl_transfer_training = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(transfer_exp_prop.iloc[0])))
    # KL[exp last training; exp pretraining]
    kl_pretraining_training = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(pretraining_exp.iloc[0])))

    gT = kl_transfer_training - kl_pretraining_training

    # filter pretraining to have the same pid as training/transfer
    pretraining_count = pretraining_count[pretraining_count["pid"].isin(training_exp.columns)]

    # now create a data object with actual strategy
    strategy_dict = {}
    strategy_dict["last_training"] = training_exp.iloc[-1].values  # get last row
    strategy_dict["exp_transfer"] = transfer_count[transfer_count["trial"] == 0]["strategy"].values  # get first column
    strategy_dict["pretraining_transfer"] = pretraining_count["strategy"].values

    # Copy pooled distribution:
    pS = copy.copy(strategy_dict)
    # Initialize permutation:
    pD = []
    # Define p (number of permutations), i.e. length of pD:
    p = 10000
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data: randomly swap cells in the columns exp_transfer and control_transfer
        # pool the values from exp_transfer and control_transfer together
        pool = list(pS["pretraining_transfer"]) + list(pS["exp_transfer"])
        random.shuffle(pool)

        # get first n items, should have same length as entries in strategy_dict["control_transfer"], i.e. 15
        pS["pretraining_transfer"] = pool[0:len(pS["pretraining_transfer"])]
        # get last n items, , should have same length as entries in strategy_dict["exp_transfer"], i.e. 18
        pS["exp_transfer"] = pool[-len(pS["exp_transfer"]):]

        # calculate the average after shuffle, 0 is control transfer, 1 is exp transfer
        average_after_shuffle = pd.DataFrame(columns=exp_training_last_trial.columns, index=[0, 1])
        # for columns in pS:
        for columns in exp_training_last_trial.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["pretraining_transfer"].count(columns) / len(pS["pretraining_transfer"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[0][columns] = proportion

        for columns in exp_training_last_trial.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["exp_transfer"].count(columns) / len(pS["exp_transfer"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[1][columns] = proportion

        # calculate KL across the shuffled, averaged values
        average_after_shuffle = average_after_shuffle.replace(0, 0.00001)
        shuffled_kl_first = np.sum(kl_div(average_after_shuffle.iloc[0], exp_training_last_trial.values))
        shuffled_kl_last = np.sum(kl_div(average_after_shuffle.iloc[1], exp_training_last_trial.values))

        pD.append(shuffled_kl_first - shuffled_kl_last)

    p_val = len(np.where(pD <= gT)[0]) / (p * len(exp_training_last_trial.columns))
    print("p value of permutation test KL3", p_val)
    return None


def kl4(pretraining_control_prop, pretraining_exp_prop, transfer_control_prop, transfer_exp_prop,
        pretraining_control_count, pretraining_exp_count, transfer_control_count, transfer_exp_count):
    # KL[pre-training exp; first transfer trial exp] > KL[pre-training control; first transfer trial control]
    # experimental pretraining, control pretraining, transfer experimental, transfer control; no training

    if 'score' in pretraining_control_prop.columns:
        pretraining_control_prop = pretraining_control_prop.drop(columns=["score"])
    if 'score' in pretraining_exp_prop.columns:
        pretraining_exp_prop = pretraining_exp_prop.drop(columns=["score"])
    if 'score' in transfer_control_prop.columns:
        transfer_control_prop = transfer_control_prop.drop(columns=["score"])
    if 'score' in transfer_exp_prop.columns:
        transfer_exp_prop = transfer_exp_prop.drop(columns=["score"])

    # miss the missing columns, i.e. strategy types for both transfer and pretraining
    if transfer_exp_prop.shape[1] != pretraining_exp_prop.shape[1]:
        missing_columns = transfer_exp_prop.columns.difference(pretraining_exp_prop.columns).tolist()
        for column in missing_columns:
            pretraining_exp_prop[column] = 0

    if transfer_exp_prop.shape[1] != pretraining_control_prop.shape[1]:
        missing_columns = transfer_exp_prop.columns.difference(pretraining_control_prop.columns).tolist()
        for column in missing_columns:
            pretraining_control_prop[column] = 0

    if transfer_control_prop.shape[1] != pretraining_control_prop.shape[1]:
        missing_columns = pretraining_control_prop.columns.difference(transfer_control_prop.columns).tolist()
        for column in missing_columns:
            transfer_control_prop[column] = 0

    pretraining_exp_prop = pretraining_exp_prop.replace(0, 0.00001)
    pretraining_control_prop = pretraining_control_prop.replace(0, 0.00001)
    transfer_exp_prop = transfer_exp_prop.replace(0, 0.00001)
    transfer_control_prop = transfer_control_prop.replace(0, 0.00001)

    # make sure all pd columns align
    pretraining_exp_prop = pretraining_exp_prop.reindex(sorted(pretraining_exp_prop.columns), axis=1)
    pretraining_control_prop = pretraining_control_prop.reindex(sorted(pretraining_control_prop.columns), axis=1)
    transfer_exp_prop = transfer_exp_prop.reindex(sorted(transfer_exp_prop.columns), axis=1)
    transfer_control_prop = transfer_control_prop.reindex(sorted(transfer_control_prop.columns), axis=1)

    # KL[pre-training exp; first transfer trial exp]
    kl_exp = np.sum(kl_div(pretraining_exp_prop.values, transfer_exp_prop.iloc[0].values))
    # KL[pre-training control; first transfer trial control]
    kl_control = np.sum(kl_div(pretraining_control_prop.values, transfer_control_prop.iloc[0].values))

    gT = kl_exp - kl_control

    # filter pid so that exp pretraining = exp transfer and control pretraining = control transfer
    pretraining_exp_count = pretraining_exp_count[pretraining_exp_count["pid"].isin(transfer_exp_count["pid"])]
    pretraining_control_count = pretraining_control_count[
        pretraining_control_count["pid"].isin(transfer_control_count["pid"])]

    # create dict with actual counts
    strategy_dict = {}
    strategy_dict["exp_pretraining"] = pretraining_exp_count.loc[(pretraining_exp_count["trial"] == 0)][
        "strategy"].values
    strategy_dict["control_pretraining"] = pretraining_control_count.loc[(pretraining_control_count["trial"] == 0)][
        "strategy"].values
    strategy_dict["exp_transfer"] = transfer_exp_count.loc[(transfer_exp_count["trial"] == 0)]["strategy"].values
    strategy_dict["control_transfer"] = transfer_control_count.loc[(transfer_control_count["trial"] == 0)][
        "strategy"].values

    pS = copy.copy(strategy_dict)

    pD = []
    # Define p (number of permutations):
    p = 10000

    for i in range(0, p):
        # Shuffle the data: randomly swap cells in the columns exp_transfer and control_transfer
        # pool the values from exp_transfer and control_transfer together
        pool = list(pS["exp_pretraining"]) + list(pS["control_pretraining"])
        random.shuffle(pool)

        # get first n items, should have same length as entries in strategy_dict["control_transfer"], i.e. 15
        pS["exp_pretraining"] = pool[0:len(pS["exp_pretraining"])]
        # get last n items, , should have same length as entries in strategy_dict["exp_transfer"], i.e. 18
        pS["control_pretraining"] = pool[-len(pS["control_pretraining"]):]

        # calculate the average after shuffle, 0 is control transfer, 1 is exp transfer
        average_after_shuffle = pd.DataFrame(columns=transfer_exp_prop.columns, index=[0, 1])
        # for columns in pS:
        for columns in transfer_exp_prop.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["exp_pretraining"].count(columns) / len(pS["exp_pretraining"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[0][columns] = proportion

        for columns in transfer_exp_prop.columns:
            # count occurence of cluster name in the column
            try:
                proportion = pS["control_pretraining"].count(columns) / len(pS["control_pretraining"])
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            average_after_shuffle.iloc[1][columns] = proportion

        # calculate KL across the shuffled, averaged values
        average_after_shuffle = average_after_shuffle.replace(0, 0.00001)
        shuffled_kl_exp = np.sum(
            kl_div(list(average_after_shuffle.iloc[0].values), list(transfer_exp_prop.iloc[0].values)))
        shuffled_kl_control = np.sum(
            kl_div(list(average_after_shuffle.iloc[1].values), list(transfer_control_prop.iloc[0].values)))

        pD.append(shuffled_kl_exp - shuffled_kl_control)

    # gT = kl_exp - kl_control, null hypothesis needs to be other way round to reject it
    # >= is correct, checked with Falk
    p_val = len(np.where(pD >= gT)[0]) / (
            p * len(
        pretraining_control_prop.columns))  # assuming number of participants  in control and exp are the same
    print("p-value of permutation test KL4", p_val)
    return None

def anvoca_preprocessing(input_dataframe, condition=""):
    # create a df with ["pid", "condition", "pretraining", "transfer"] and pid as index
    # df = pd.DataFrame(columns=["pid", "condition", "pretraining", "transfer"])

    df = input_dataframe[["pid", "block", "score"]]
    df_grouped = df.groupby(by=["pid", "block"]).mean("score")
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped[df_grouped["block"] != "training"]
    df_grouped = df_grouped.fillna(0)
    df_grouped = df_grouped.pivot(index="pid", columns='block', values='score')
    df_grouped["condition"] = condition
    return df_grouped



if __name__ == "__main__":

    ### Select experiment
    experiment = "roadtrip"
    number_of_test_trials = 15
    pre_training = False
    filters = True
    if pre_training:
        number_of_training_trials = 35
    else:
        number_of_training_trials = 10


    ### load data
    if experiment == "mortgage":
        exp_data = pd.read_csv('data/human/exp_mortgage/mouselab-mdp.csv')
        control_data = pd.read_csv('data/human/control_mortgage/mouselab-mdp.csv')
    if experiment == "roadtrip":
        exp_data = pd.read_csv('data/human/exp_roadtrip/mouselab-mdp.csv')
        control_data = pd.read_csv('data/human/control_roadtrip/mouselab-mdp.csv')

    ##### CREATE DATA
    non_grouped_control_df, grouped_control_df, actual_count_control_transfer = create_trial_df_for_each_participant_group(
        control_data, experiment, pre_training=False)
    non_grouped_exp_df, grouped_exp_df, actual_count_exp_transfer = create_trial_df_for_each_participant_group(
        exp_data, experiment, pre_training=False)
    if pre_training:
        non_grouped_control_pretraining_df, grouped_control_pretraining_df, actual_count_control_pretraining_transfer = create_trial_df_for_each_participant_group(
            control_data, experiment, pre_training=True)
        non_grouped_exp_pretraining_df, grouped_exp_pretraining_df, actual_count_exp_pretraining_transfer = create_trial_df_for_each_participant_group(
            exp_data, experiment, pre_training=True)

    ### filter non_grouped_df for participants who changed their strategy during training (exp only)
    filtered_non_grouped_exp_df = filter_pid_who_changed_strategy_during_training(non_grouped_exp_df, filters)

    ##### ANALYSIS 1
    ### find out how many people used the same strategy type at the end of training and beginning of transfer, only for exp condition
    training_exp, transfer_exp, transfer_proportion, actual_number_of_transfer, total_number_of_filtered_participants = check_for_same_strategy_before_after_transfer(
        filters, "exp", experiment=experiment, transferdata=filtered_non_grouped_exp_df)

    ### chi-squared test for transfer proportion
    # value1 = [actual_number_of_transfer, total_number_of_filtered_participants - actual_number_of_transfer]
    # value2 = [0.2 * total_number_of_filtered_participants, 0.8 * total_number_of_filtered_participants]
    # m = np.row_stack([value1, value2])
    # res = stats.fisher_test(m, simulate_p_value=True)
    # print(f"Fisher exact test to test for transfer proportion: p={res[0][0]:.3f}")
    #
    ## The chi-square test tests the null hypothesis that the categorical data has the given frequencies.
    print("Chi-squared test to test for transfer proportion: ",
          {chisquare(
              [(actual_number_of_transfer), ((total_number_of_filtered_participants - actual_number_of_transfer))],
              f_exp=[(1/6) * total_number_of_filtered_participants, (5/6) * total_number_of_filtered_participants])})
    ##### ANALYSIS 2
    ##### CREATE PLOTS
    # create_plots_between_adaptive_maladaptive(grouped_control_df, grouped_exp_df_all, grouped_exp_df_adaptive, grouped_exp_df_maladaptive,
    #              number_of_trials=15)
    # if pre_training is False:
    #     grouped_control_pretraining_df, grouped_exp_pretraining_df = None, None
    # create_plots_training_test(training_exp, grouped_control_df, grouped_exp_df, grouped_control_pretraining_df,
    #                            grouped_exp_pretraining_df)
    # plot_strategy_distribution(training_exp, grouped_control_df, grouped_exp_df, pre_training,
    #                            grouped_control_pretraining_df, grouped_exp_pretraining_df)
    # plot_cluster_proportions(training_exp, transfer_exp)
    # kl1(training_exp, transfer_exp, grouped_exp_df)
    # kl2(training_exp, transfer_exp, grouped_exp_df, actual_count_control_transfer, non_grouped_control_df)
    # kl3(training_exp, grouped_exp_df, grouped_exp_pretraining_df, non_grouped_exp_pretraining_df, transfer_exp)
    # if pre_training:
    #     kl4(grouped_exp_pretraining_df, grouped_control_pretraining_df, grouped_control_df, grouped_exp_df,
    #         non_grouped_control_pretraining_df, non_grouped_exp_pretraining_df, non_grouped_control_df, non_grouped_exp_df)

    # Fisher test to test between two conditions of first trial control and experimental
    if actual_count_control_transfer.shape[1] != actual_count_exp_transfer.shape[1]:
        missing_columns = actual_count_exp_transfer.columns.difference(
            actual_count_control_transfer.columns).tolist()
        for column in missing_columns:
            actual_count_control_transfer[column] = 0

    value1 = actual_count_control_transfer.iloc[0].values
    # if experiment == "mortgage":
    # value1 = np.pad(value1, (0, 1), 'constant')  # todo: CHECK WHETHER IT IS OKAY TO ADD 0 IN THE END
    value2 = actual_count_exp_transfer.iloc[0].values
    m = np.row_stack([value1, value2])
    m_prop = m / 36
    res = stats.fisher_test(m, simulate_p_value=True)
    print(f"Fisher exact test of distribution of first trial between control and experimental: p={res[0][0]:.3f}")

    # if pre_training:
    #     ## Fisher test to test between pretraining and first transfer in experimental condition
    #     ##add the missing columsn
    #     if actual_count_exp_pretraining_transfer.shape[1] != actual_count_exp_transfer.shape[1]:
    #         missing_columns = actual_count_exp_transfer.columns.difference(
    #             actual_count_exp_pretraining_transfer.columns).tolist()
    #         for column in missing_columns:
    #             actual_count_exp_pretraining_transfer[column] = 0
    #
    #     # reindex, i.e. reorder the columns
    #     actual_count_exp_pretraining_transfer = actual_count_exp_pretraining_transfer.reindex(
    #         sorted(actual_count_exp_pretraining_transfer.columns), axis=1)
    #     actual_count_exp_transfer = actual_count_exp_transfer.reindex(sorted(actual_count_exp_transfer.columns), axis=1)
    #
    #     value1 = actual_count_exp_pretraining_transfer.values
    #     value2 = actual_count_exp_transfer.iloc[0].values
    #     m = np.row_stack([value1, value2])
    #     res = stats.fisher_test(m, simulate_p_value=True)
    #     print(
    #         f"Fisher exact test of distribution of pretraining and transfer in experimental condition: p={res[0][0]:.3f}")
    #
    # # ### Wilcoxon test to test between scores
    # # ### The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution.
    print("Mean score control", grouped_control_df["score"].mean())
    print("Mean score exp", grouped_exp_df["score"].mean())
    print("SD score control", grouped_control_df["score"].std())
    print("SD score exp", grouped_exp_df["score"].std())
    # print("Wilcoxon rank sum test of score between control and experimental: ",
    #       ranksums(grouped_control_df["score"][:5], grouped_exp_df["score"][:5], alternative="less"))
    print("Wilcoxon rank sum test of score between control and experimental: ",
          ranksums(grouped_exp_df["score"], grouped_control_df["score"], alternative="greater"))

    # ### trend test
    # print("Kendall Trend test for the control group")
    # trend_tests(grouped_control_df)
    # print("Kendall Trend test for the experimental group")
    # trend_tests(grouped_exp_df)
    #
    # ### Fisher test between first training and last training
    # exp_training_grouped = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
    #                                              "Middle planning", "Frugal planning", "Other planning"], index=[0, 1])
    # for columns in exp_training_grouped:
    #     exp_training_grouped[columns].iloc[0] = list(training_exp.iloc[1]).count(columns)
    #     exp_training_grouped[columns].iloc[1] = list(training_exp.iloc[-1]).count(columns)
    # value1 = exp_training_grouped.iloc[0].values.astype(float)
    # value2 = exp_training_grouped.iloc[1].values.astype(float)
    # m = np.row_stack([value1, value2])
    # res = stats.fisher_test(m, simulate_p_value=True)
    # print(f"Fisher exact test of distribution of first training and last training: p={res[0][0]:.3f}")
    # # print("Chi-squared test of distribution of first training and last training: ", {chisquare( [value1],f_exp=value2)})


    ### ANCOVA
    # r = robjects.r
    #
    # x = robjects.IntVector([1, 0])
    # y = robjects.FloatVector([1, 0])
    # g = r.matrix(r.rnorm(100), ncol=2)
    # # print(sm)
    # ancova_res = sm.sm_ancova(x, y, g)
    # print(ancova_res)
    if pre_training:

        # control_data["condition"] = "control"
        # exp_data["condition"] = "exp"

        control_data = anvoca_preprocessing(input_dataframe=control_data, condition="control")
        exp_data = anvoca_preprocessing(input_dataframe=exp_data, condition="exp")

        df_final = pd.concat([control_data, exp_data])
        df_final.to_csv(f"{experiment}_anvoca.csv")

        # print(df_final)

        control_pretraining_score = df_final[df_final["condition"] == "control"]["pretraining"]
        control_test_score = df_final[df_final["condition"] == "control"]["test"]
        exp_pretraining_score = df_final[df_final["condition"] == "exp"]["pretraining"]
        exp_test_score = df_final[df_final["condition"] == "exp"]["test"]

        # check for normality of the score
        from scipy import stats

        k2, p = stats.normaltest(control_pretraining_score)
        print(f"Normality check control_pretraining_score: S= {k2}, p = {p}")
        k2, p = stats.normaltest(control_test_score)
        print(f"Normality check control_test_score: S= {k2},p = {p}")
        k2, p = stats.normaltest(exp_pretraining_score)
        print(f"Normality check exp_pretraining_score: S= {k2},p = {p}")
        k2, p = stats.normaltest(exp_test_score)
        print(f"Normality check exp_test_score: S= {k2}, p = {p}")

        # check for homegeneity
        # stat, p = stats.levene(control_test_score, control_pretraining_score)
        # print(f"homogeneity control:  p={p}")
        # stat, p = stats.levene(exp_test_score, exp_pretraining_score)
        # print(f"homogeneity exp:  p={p}")

        stat, p = stats.levene(exp_test_score, control_test_score)
        print(f"homogeneity between exp and control test score:  p={p}")

        # anvoca
        from pingouin import ancova
        res = ancova(data=df_final, dv='test', covar='pretraining', between='condition')
        print(res)