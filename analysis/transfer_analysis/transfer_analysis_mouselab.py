import pandas as pd
import os
import copy
import random
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import chisquare, ranksums, ttest_rel, chi2_contingency
from scipy.special import kl_div
import json

from mcl_toolbox.utils.learning_utils import pickle_load, create_dir

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
stats = importr("stats")

random.seed(0)


def create_strategy_proportion_plots(control_data, exp_data, exp_adaptive_data, exp_maladaptive_data):
    x = range(0, len(control_data))
    # plot the proportion of adaptive strategies
    proportion_exp_std = exp_data["proportion_of_adaptive_strategies"].std()
    proportion_control_std = control_data["proportion_of_adaptive_strategies"].std()
    plt.plot(exp_data.index, exp_data["proportion_of_adaptive_strategies"], label="Experimental")
    plt.plot(control_data.index, control_data["proportion_of_adaptive_strategies"], label="Control")
    if exp_adaptive_data is not None:
        plt.plot(exp_adaptive_data.index, exp_adaptive_data["proportion_of_adaptive_strategies"],
                 label="Experimental adaptive")
    if exp_maladaptive_data is not None:
        plt.plot(exp_maladaptive_data.index, exp_maladaptive_data["proportion_of_adaptive_strategies"],
                 label="Experimental maladaptive")
    plt.fill_between(x, control_data["proportion_of_adaptive_strategies"] - proportion_control_std,
                     control_data["proportion_of_adaptive_strategies"] + proportion_control_std, alpha=0.5,
                     label='SD of adaptive strategies')
    plt.fill_between(x, exp_data["proportion_of_adaptive_strategies"] - proportion_exp_std,
                     exp_data["proportion_of_adaptive_strategies"] + proportion_exp_std, alpha=0.5,
                     label='SD of adaptive strategies')
    plt.legend(fontsize='xx-small')
    plt.title('Proportion of adaptive strategies')
    create_dir("results/cm/plots/mouselab")
    plt.savefig(
        f"results/cm/plots/mouselab/performance.png",
        bbox_inches="tight",
    )
    plt.close()

    # plot score
    proportion_control_score_std = control_data["score"].std()
    proportion_exp_score_std = exp_data["score"].std()
    plt.plot(exp_data.index, exp_data["score"], label="Experimental")
    plt.plot(control_data.index, control_data["score"], label="Control")
    if exp_adaptive_data is not None:
        plt.plot(exp_adaptive_data.index, exp_adaptive_data["score"], label="Experimental adaptive")
    if exp_maladaptive_data is not None:
        plt.plot(exp_maladaptive_data.index, exp_maladaptive_data["score"],
                 label="Experimental maladaptive")
    plt.fill_between(x, control_data["score"] - proportion_control_score_std,
                     control_data["score"] + proportion_control_score_std, alpha=0.5,
                     label='SD of adaptive strategies')
    plt.fill_between(x, exp_data["score"] - proportion_exp_score_std,
                     exp_data["score"] + proportion_exp_score_std, alpha=0.5,
                     label='SD of adaptive strategies')
    plt.legend(fontsize='xx-small')
    plt.title('Score')
    plt.savefig(
        f"results/cm/plots/mouselab/score.png",
        bbox_inches="tight",
    )


def create_cluster_proportion_plots(control_data, exp_data, exp_adaptive_data, exp_maladaptive_data):
    # plots grouped by strategy clusters
    for columns in exp_data:
        plt.plot(exp_data.index, exp_data[columns], label="Experimental")
        plt.plot(control_data.index, control_data[columns], label="Control")
        plt.plot(exp_adaptive_data.index, exp_adaptive_data[columns], label="Experimental adaptive")
        plt.plot(exp_maladaptive_data.index, exp_maladaptive_data[columns], label="Experimental maladaptive")
        plt.legend(fontsize='xx-small')
        plt.title(columns)
        plt.savefig(f"results/cm/plots/mouselab/{columns}.png", bbox_inches="tight", )
        plt.close()

    # plots grouped by group (control ,experimental, etc)
    control_data.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Control")
    plt.savefig(f"results/cm/plots/mouselab/control.png", bbox_inches="tight", )
    plt.close()

    exp_data.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental")
    plt.savefig(f"results/cm/plots/mouselab/experimental.png", bbox_inches="tight", )
    plt.close()

    exp_adaptive_data.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental adaptive")
    plt.savefig(f"results/cm/plots/mouselab/experimental_adaptive.png", bbox_inches="tight", )

    exp_maladaptive_data.plot()
    plt.legend(fontsize='xx-small')
    plt.title("Experimental maladaptive")
    plt.savefig(f"results/cm/plots/mouselab/experimental_maladaptive.png", bbox_inches="tight", )


def create_strategy_proportions(participant_strategies_, participant_score_data, control_condition):
    """
    group according to adaptive, maladaptive and other strategies
    Args:
        participant_strategies_: df of participants and their strategies
        strategy_scores: dict of strategy and their scores

    Returns:

    """
    # load strategies and their scores for large mouselab
    strategy_scores = pd.read_pickle("results/cm/strategy_scores/large_mouselab_strategy_scores.pkl")

    # group strategies into adaptive, maladaptive and other according to score
    ### clustering of the strategy score:
    # create df with columns strategy, value, label
    strategy_df = pd.DataFrame(columns=["strategy", "label"])

    strategy_values_list = np.array(list(strategy_scores.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(strategy_values_list)
    strategy_df["label"] = kmeans.labels_
    strategy_df["strategy"] = strategy_scores.keys()

    ## relabel the cluster centers
    cluster_centers = pd.Series(kmeans.cluster_centers_.flatten())
    cluster_centers = cluster_centers.sort_values()
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[0]), "maladaptive_strategies")
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[1]), "other_strategies")
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[2]), "adaptive_strategies")

    adaptive_strategies = list(strategy_df[strategy_df['label'] == "adaptive_strategies"]["strategy"])
    maladaptive_strategies = list(strategy_df[strategy_df['label'] == "maladaptive_strategies"]["strategy"])
    other_strategies = list(strategy_df[strategy_df['label'] == "other_strategies"]["strategy"])

    # match adaptive, maladaptive and other to the participants
    # todo: change calculate strategy score to start from 1 as well
    participant_strategies_ = participant_strategies_ - 1  # because when calculating strategy score, strategy start at 0 but here it starts at 1
    participant_strategies_ = participant_strategies_.replace(adaptive_strategies, "adaptive_strategies")
    participant_strategies_ = participant_strategies_.replace(maladaptive_strategies, "maladaptive_strategies")
    participant_strategies_ = participant_strategies_.replace(other_strategies, "other_strategies")

    # calculate the PROPORTION of adaptive strategies
    adaptive_proportion_trialwise = []
    for column in participant_strategies_:
        proportion_adaptive_strategies_trialwise = len(
            participant_strategies_[participant_strategies_[column] == "adaptive_strategies"]) / len(
            participant_strategies_[column])
        adaptive_proportion_trialwise.append(proportion_adaptive_strategies_trialwise)

    proportion_df = pd.DataFrame()
    number_of_trials = len(participant_strategies_.columns)
    proportion_df["trial"] = list(range(0, number_of_trials))
    proportion_df["proportion_of_adaptive_strategies"] = adaptive_proportion_trialwise

    # filter the participant_SCORE_data to only block=test
    participant_score_data = participant_score_data.loc[participant_score_data["block"] == "test"]
    participant_score_data = participant_score_data[
        participant_score_data['pid'].isin(participant_strategies_.index.values.tolist())]

    # good_participants_list = remove_no_click_participants(control_condition)
    # participant_score_data = participant_score_data[participant_score_data['pid'].isin(good_participants_list)]

    # need to reset trial_index so that it counts from 1:number of trials
    trials = list(range(0, number_of_trials)) * participant_strategies_.shape[0]  # = number of participants
    participant_score_data["trial_index"] = trials

    # create averages of score
    grouped_df = participant_score_data.groupby(["trial_index"]).mean()
    proportion_df["score"] = grouped_df["score"]

    return proportion_df


def create_cluster_df(data, filters, pretraining=False, transfer=True):
    """

    Args:
        data:

    Returns: data: a df with pid as row and trials and trials and columns, each cell contains the strategy type used
    cluster_proportion: percentage of strategy on trial basis

    """

    # cluster the participant strategies according to clusters
    cluster_mapping_dict = pd.read_pickle("../../mcl_toolbox/data/kl_cluster_map.pkl")

    # replace strategy by cluster
    data = data.replace(cluster_mapping_dict)

    # cluster the clusters
    cluster_mapping = {"Goal setting": [9],
                       "Backward planning": [1, 6],
                       "Forward planning": [2, 4, 8, 11, 12],
                       "Middle planning": [3],
                       "Frugal planning": [7, 10],
                       "Local search": [5],
                       "Other planning": [13]}

    # reshape the dict by switching dict and value
    cluster_mapping_temp_dict = {}
    for k, v in cluster_mapping.items():
        for item in v:
            cluster_mapping_temp_dict[item] = k
    data = data.replace(cluster_mapping_temp_dict)

    # remove participants who did not change their strategy during training
    # data.shape[1] > 1 ensures that pre-training trial will not be removed as there is only 1 trial aka 1 strategy
    if transfer is False:
        good_participants_list = []
        for index, rows in data.iterrows():
            if len(rows.unique()) > 1:
                good_participants_list.append(index)
        data_filtered = data[data.index.isin(good_participants_list)]
    else:
        data_filtered = data
    # if we want to use all participants
    if filters is False:
        data_filtered = data

    if len(
            data.columns) == 1:  # ensures that pre-training trial will not be removed as there is only 1 trial aka 1 strategy
        data_filtered = data

    # create proportions
    cluster_proportion = pd.DataFrame(columns=list(cluster_mapping.keys()), index=range(len(data_filtered.columns)))
    for trial in data_filtered:
        for strategy_cluster_names in cluster_proportion:
            # count occurence of cluster name in the column (i.e. trial)
            try:
                proportion = data_filtered[trial].value_counts()[strategy_cluster_names] / len(data_filtered.index)
            except:  # strategy not found, i.e. not used
                proportion = 0
            # assign the value to the df
            cluster_proportion.iloc[trial][strategy_cluster_names] = proportion

    # create actual counts
    actual_count = pd.DataFrame(columns=list(cluster_mapping.keys()), index=range(len(data_filtered.columns)))
    for trial in data_filtered:
        for strategy_cluster_names in actual_count:
            # count occurence of cluster name in the column (i.e. trial)
            try:
                counts = data_filtered[trial].value_counts()[strategy_cluster_names]
            except:  # strategy not found, i.e. not used
                counts = 0
            # assign the value to the df
            actual_count.iloc[trial][strategy_cluster_names] = counts

    if transfer is False:
        return data_filtered, cluster_proportion, good_participants_list, actual_count
    else:
        return data_filtered, cluster_proportion, actual_count


def create_strategy_score_df(participant_data, participant_score_data, type, control_condition=True):
    """
    Create several df
    Args:
        participant_data:
        participant_score_data:
        type:
        control_condition:

    Returns:
        if cluster and control condition:
            cluster_proportion_all = df with strategy proportions;
            actual_count = df with count of strategies


    """
    participant_data = pd.DataFrame.from_dict(participant_data, orient='index')

    # filter out participants who did not change their strategy in the last 3 trials of training
    if type == "strategy":
        strategy_proportion_all = create_strategy_proportions(participant_data, participant_score_data,
                                                              control_condition)
        return strategy_proportion_all

    else:
        data, cluster_proportion_all, good_participants_list, actual_count = create_cluster_df(participant_data,
                                                                                               filters,
                                                                                               transfer=False)
        return cluster_proportion_all, actual_count, good_participants_list, data


def trend_tests(grouped_df):
    ### Mann Kendall test for trend
    result_clicks = mk.original_test(grouped_df["Goal setting"])
    # result_score = mk.original_test(grouped_df["score"])
    print("Mann Kendall test of proportion of adaptive strategies", result_clicks)
    # print("Mann Kendall test of score", result_score)


def remove_no_click_participants(control_condition):
    # remove participants with 0 clicks
    if control_condition:
        participant_clicks = pd.read_pickle("results/cm/inferred_strategies/T1.1_test/clicks.pkl")
        participant_clicks = pd.DataFrame.from_dict(participant_clicks)
    else:
        participant_clicks = pd.read_pickle("results/cm/inferred_strategies/exp_mouselab_test/clicks.pkl")
        participant_clicks = pd.DataFrame.from_dict(participant_clicks)

    # replace the actual click location with 0 when there are no clicks, 1 when there was a click
    participant_clicks.replace(1, 0)  # 1 in the click dict means no click
    participant_clicks[participant_clicks != 0] = 1  # replace everything else with 1
    good_participants_list = []
    for column in participant_clicks:
        if participant_clicks[column].sum() != 0:
            good_participants_list.append(column)
        else:
            print(f"PID {column} removed due to no clicks")

    return good_participants_list


def check_for_same_strategy_before_after_transfer(exp_large_mouselab_test_strategies, exp_training_strategies, filters):
    """
    This function checks how many people (in percentage) used the same stratety type (e.g. goal setting) before and after transfer,
    that is the last training trial and first transfer trial

    Args:
        exp_large_mouselab_test_strategies:
        exp_training_strategies:

    Returns:

    """
    exp_large_mouselab_test_strategies = pd.DataFrame.from_dict(exp_large_mouselab_test_strategies, orient='index')
    exp_training_strategies = pd.DataFrame.from_dict(exp_training_strategies, orient='index')

    # # get first transfer trial
    pid_level_data_transfer, _, actual_count_test = create_cluster_df(exp_large_mouselab_test_strategies, filters,
                                                                      transfer=True)
    # get last strategy in training
    pid_level_data_training, _, _, actual_count_training = create_cluster_df(exp_training_strategies, filters,
                                                                             transfer=False)

    # get intersection of pid between training and transfer data (some training pid were removed)
    pid_list = list(set(pid_level_data_transfer.index) & set(pid_level_data_training.index))
    pid_level_data_transfer = pid_level_data_transfer[pid_level_data_transfer.index.isin(pid_list)]

    # filter out participants who did not change their strategy in the last 3 trials of training
    good_participants_list = []
    for index, row in pid_level_data_training.iterrows():
        if len(row[-3:].unique()) == 1:
            good_participants_list.append(index)

    if filters is True:
        pid_level_data_transfer = pid_level_data_transfer[pid_level_data_transfer.index.isin(good_participants_list)]
        pid_level_data_training = pid_level_data_training[pid_level_data_training.index.isin(good_participants_list)]

    # get last row of
    print(pid_level_data_training.iloc[:, -1])
    print(pid_level_data_training.iloc[:, -1] = "NaN")
    print(pid_level_data_training.iloc[(pid_level_data_training.iloc[:, -1] == "nan"), -1])
    pid_level_data_training.iloc[(pid_level_data_training.iloc[:, -1] == "nan"), -1] = pid_level_data_training.iloc[:,
                                                                                       9]
    # compare first column from transfer to last column in training
    comparison = np.where(pid_level_data_transfer.iloc[:, 0] == pid_level_data_training.iloc[:, -1], 1,
                          0)  # 1 is true, 0 is false
    percentage_of_same_strategy_types = np.sum(comparison) / len(comparison)
    print("Actual number of PID using the same strategy type before and after transfer: ",
          np.sum(comparison), len(comparison))
    print("Percentage of PID using the same strategy type before and after transfer: ",
          percentage_of_same_strategy_types)
    return pid_level_data_training, pid_level_data_transfer, percentage_of_same_strategy_types, np.sum(comparison), len(
        comparison)


def create_plots_training_test(training_exp, control_df, transfer_exp, pre_training, pretraining_control,
                               pretraining_exp):
    # need proportion of goal setting for training, control, exp, pretraining control, pretraining exp
    if pre_training:
        x_range_pretraining = 0
        x_range_exp = range(1, (training_exp.shape[1] + 1) + len(transfer_exp))
        x_range_control = range(training_exp.shape[1] + 1, training_exp.shape[1] + len(transfer_exp) + 1)
    else:
        x_range_exp = range(0, (training_exp.shape[1]) + len(transfer_exp))
        x_range_control = range(training_exp.shape[1], training_exp.shape[1] + len(transfer_exp))

    def replace_strategies(data):
        data = data.replace("Goal setting", 1)
        data = data.replace("Frugal planning", 0)
        data = data.replace("Forward planning", 0)
        data = data.replace("Middle planning", 0)
        data = data.replace("Backward planning", 0)
        data = data.replace("Local search", 0)
        data = data.replace("Other planning", 0)
        return data

    # calculate the proportions
    training_replaced = replace_strategies(training_exp)
    training_proportion = training_replaced.sum(axis=0)
    training_proportion = training_proportion / len(training_replaced)
    # training_proportion = training_replaced.div(len(training_replaced.columns), axis=1).sum(axis=0)
    # training_proportion = training_proportion.fillna(0)

    control = list(control_df["Goal setting"])
    experimental = list(training_proportion) + list(transfer_exp["Goal setting"])

    if pre_training:
        plt.plot(x_range_pretraining, pretraining_control["Goal setting"], 'ro', label="Pretraining Control")
        plt.plot(x_range_pretraining, pretraining_exp["Goal setting"], 'go', label="Pretraining Experimental")
    plt.plot(x_range_control, control, label="Control")
    plt.plot(x_range_exp, experimental, label="Experimental")
    plt.axvline(x=training_exp.shape[1])  # TODO if pretraining, add 1
    plt.title("Proportion of Goal setting strategy")
    plt.legend()
    # plt.savefig(
    #     f"results/cm/plots/mouselab_training_transfer.png",
    #     bbox_inches="tight",
    # )

    plt.show()
    plt.close()
    return None


def plot_strategy_distribution(training_exp, control_df, transfer_exp, pretraining, pretraining_control,
                               pretraining_exp):
    # need: first training, last training, first exp transfer, first control transfer
    # optional pretraining exp, pretraining control
    if pretraining:
        x = ["exp pre", "control pre", "exp first train.", "exp last train.", "exp first trans.",
             "control first trans."]
        df_training = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                            "Middle planning", "Frugal planning", "Local search", "Other planning"],
                                   index=range(0, 35))
    else:
        x = ["exp first training", "exp last training", "exp first transfer", "control first transfer"]
        df_training = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                            "Middle planning", "Frugal planning", "Local search", "Other planning"],
                                   index=range(0, 10))
    x_axis = np.arange(len(x))

    for trials in training_exp:  # columns
        for columns in df_training:
            df_training[columns].iloc[trials] = len(training_exp[training_exp[trials] == columns])

    training_proportion = df_training.div(df_training.sum(axis=1), axis=0)

    # need them as as df, therefore :1 to get the first row
    training_first = training_proportion.iloc[:1]
    training_last = training_proportion.iloc[-1:]
    exp_first_transfer = transfer_exp.iloc[:1]
    control_first_transfer = control_df.iloc[:1]

    # concatenate all
    df = pd.concat([training_first, training_last, exp_first_transfer, control_first_transfer])
    if pre_training:
        df = pd.concat([pretraining_control, pretraining_exp, training_first, training_last, exp_first_transfer,
                        control_first_transfer])

    goal_setting = df["Goal setting"]
    backward_planning = df["Backward planning"]
    forward_planning = df["Forward planning"]
    middle_planning = df["Middle planning"]
    frugal_planning = df["Frugal planning"]
    local_search = df["Local search"]
    other_planning = df["Other planning"]

    width = 0.13
    plt.bar(x_axis - width * 3, goal_setting, width=0.13, align='center', label="Goal setting")
    plt.bar(x_axis - width * 2, backward_planning, width=0.13, align='center', label="Backward planning")
    plt.bar(x_axis - width, forward_planning, width=0.13, align='center', label="Forward planning")
    plt.bar(x_axis, middle_planning, width=0.13, align='center', label="Middle planning")
    plt.bar(x_axis + width, frugal_planning, width=0.13, align='center', label="Frugal planning")
    plt.bar(x_axis + width * 2, local_search, width=0.13, align='center', label="Local search")
    plt.bar(x_axis + width * 3, other_planning, width=0.13, align='center', label="Other planning")
    plt.axvline(0.48)
    plt.axvline(1.48)
    plt.axvline(2.50)
    if pre_training:
        plt.axvline(3.45)
        plt.axvline(4.45)

    plt.legend(fontsize=10)
    plt.xticks(x_axis, x, fontsize=10)
    # plt.savefig(f"results/cm/plots/mouselab_strategy_distribution_with_local_search.png", bbox_inches="tight",)

    plt.show()
    plt.close()
    return None


def plot_cluster_proportions(training_exp, transfer_exp):
    # exp only

    # need training strategy proportions, the input are actuals
    df_training = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                        "Middle planning", "Frugal planning", "Local search", "Other planning"],
                               index=range(0, 10))

    for trials in training_exp:  # columns
        for columns in df_training:
            df_training[columns].iloc[trials] = len(training_exp[training_exp[trials] == columns])

    df_training = df_training.div(df_training.sum(axis=1), axis=0)

    df_transfer = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                        "Middle planning", "Frugal planning", "Local search", "Other planning"],
                               index=range(0, 15))

    # need transfer strategy proportions, the input are actuals
    for trials in transfer_exp:  # columns
        for columns in df_transfer:
            df_transfer[columns].iloc[trials] = len(transfer_exp[transfer_exp[trials] == columns])

    df_transfer = df_transfer.div(df_transfer.sum(axis=1), axis=0)

    # fuse both pd into one
    frames = [df_training, df_transfer]
    df = pd.concat(frames)

    x_range = range(0, training_exp.shape[1] + transfer_exp.shape[1])
    plt.plot(x_range, df)
    # plt.plot(x_range_transfer, df_transfer)
    plt.axvline(x=training_exp.shape[1])
    # plt.title("Proportion of Goal setting strategy")
    plt.legend(df.columns, loc='upper right', prop={'size': 12})
    # plt.savefig(
    #     f"results/cm/plots/mouselab_strategy_proportions_training_transfer.png",
    #     bbox_inches="tight",
    # )
    plt.show()
    plt.close()

    return None


def kl1(training, transfer, exp_data):
    """
    KL[experimental first transfer trial; experimental first training trial] > KL[experimental first transfer trial; experimental last training trial].
    Experimental condition only
    Args:
        training:
        transfer:
        exp_data:

    Returns:

    """
    # need strategy distribution of the first training trial, last training trial and first transfer trial
    training_first_last_trials = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                       "Middle planning", "Frugal planning", "Local search",
                                                       "Other planning"])

    # get proportion of first and last row of training
    for columns in training_first_last_trials:
        # count occurrence of training_df columns, i.e. the strategy types
        first_trial_proportion = list(training.iloc[1]).count(columns) / len(training.iloc[1])
        last_trial_proportion = list(training.iloc[-1]).count(columns) / len(training.iloc[-1])
        training_first_last_trials[columns] = [first_trial_proportion, last_trial_proportion]

    # drop score
    # exp_data = exp_data.drop(columns="score")
    exp_data.rename(columns={"Other strategy": "Other planning"})

    # add a very small value to the 0 in exp_data and training_first_last_trials
    exp_data = exp_data.replace(0, 0.00001)
    training_first_last_trials = training_first_last_trials.replace(0, 0.00001)

    first_training = np.sum(kl_div(list(exp_data.iloc[0]), list(training_first_last_trials.iloc[0])))
    last_training = np.sum(kl_div(list(exp_data.iloc[0]), list(training_first_last_trials.iloc[-1])))

    gT = first_training - last_training

    # create df with columns containing the strategy used by PID for first, last training trial and first transfer trial
    strategy_table = pd.DataFrame(columns=["first_training", "last_training", "first_transfer"], index=training.index)
    for index, row in strategy_table.iterrows():
        strategy_table.loc[index]["first_training"] = training.loc[index][0]
        strategy_table.loc[index]["last_training"] = training.loc[index][9]
        strategy_table.loc[index]["first_transfer"] = transfer.loc[index][0]

    # replace the strategy string names by distributions ["Goal setting", "Backward planning", "Forward planning",
    #                                                        "Middle planning", "Frugal planning", "Other planning"]

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
    print("p value of permutation test KL1 ", p_val)

    return None


def kl2(training_exp, transfer_exp, exp_data, control_transfer_count,
        control_data):
    """
    KL[experimental last training trial; experimental first transfer trial] < KL[experimental last training trial; control first transfer trial].

    Args:
        training_exp: df with pid as columns and trials as rows. Each cell contains the strategy used
        transfer_exp: ungrouped df
        exp_data: df with strategy as columns, trials as rows. Each cell contains the frequency of the strategy
        control_transfer: df with strategy as columns, trials as rows. Each cell contains the COUNT of the strategy

    Returns:

    """
    # get exp training last trial
    exp_training_last_trial = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                    "Middle planning", "Frugal planning", "Local search",
                                                    "Other planning"])
    for columns in exp_training_last_trial:
        exp_training_last_trial[columns] = [list(training_exp.iloc[-1]).count(columns) / len(training_exp.iloc[-1])]

    # get control_transfer proportions
    control_transfer_proportions = control_transfer_count.div(control_transfer_count.sum(axis=1), axis=0)

    # drop score
    exp_data.rename(columns={"Other strategy": "Other planning"})

    # add a very small value to the 0 in exp_data and training_first_last_trials
    exp_training_last_trial = exp_training_last_trial.replace(0, 0.00001)
    exp_data = exp_data.replace(0, 0.00001)
    control_transfer_proportions = control_transfer_proportions.replace(0, 0.00001)

    # kl[exp last training; exp first transfer]
    kl_exp_training_exp_transfer = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(exp_data.iloc[0])))
    # kl[exp last training; control first transfer]
    kl_exp_training_control_transfer = np.sum(
        kl_div(list(exp_training_last_trial.iloc[0]), list(control_transfer_proportions.iloc[0])))

    gT = kl_exp_training_exp_transfer - kl_exp_training_control_transfer

    # create dict with keys "control_transfer", "exp_transfer", "first_training"
    training_exp = training_exp.T
    strategy_dict = {}
    strategy_dict["last_training"] = training_exp.iloc[-1].values  # get last row
    strategy_dict["exp_transfer"] = transfer_exp.iloc[:, 0].values
    strategy_dict["control_transfer"] = control_data.iloc[:, 0].values

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

    p_val = len(np.where(pD >= gT)[0]) / (p * len(exp_data.columns))
    print("p value of permutation test KL2", p_val)

    return None


# def kl3(training_exp, transfer_exp, pretraining_exp, exp_data):
def kl3(training_exp, transfer_exp_prop, pretraining_exp, pretraining_count, transfer_count):
    """
    KL[exp transfer; exp last training] < KL[exp pretraining; exp last training]

    Args:
        training_exp:
        transfer_exp:
        pretraining_exp:
        exp_data:

    Returns:

    """

    # get exp training last trial and proportions
    exp_training_last_trial = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                    "Middle planning", "Frugal planning", "Other planning"])
    for columns in exp_training_last_trial:
        exp_training_last_trial[columns] = [list(training_exp.iloc[-1]).count(columns) / len(training_exp.iloc[-1])]

    # drop score
    transfer_exp_prop.rename(columns={"Other strategy": "Other planning"})

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
    transfer_exp_prop = transfer_exp_prop.replace(0, 0.00001)
    pretraining_exp = pretraining_exp.replace(0, 0.00001)

    # KL[exp last training; exp first transfer]
    kl_transfer_training = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(transfer_exp_prop.iloc[0])))
    # KL[exp last training; exp pretraining]
    kl_pretraining_training = np.sum(kl_div(list(exp_training_last_trial.iloc[0]), list(pretraining_exp.iloc[0])))

    gT = kl_transfer_training - kl_pretraining_training

    # filter pretraining to have the same pid as training/transfer
    pretraining_count = pretraining_count[pretraining_count.index.isin(training_exp.index)]

    # now need the actual counts
    strategy_dict = {}
    strategy_dict["last_training"] = training_exp.iloc[:, -1].values  # get last column
    strategy_dict["exp_transfer"] = transfer_count.iloc[:, 0].values  # get first column
    strategy_dict["pretraining_transfer"] = pretraining_count.values

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
        shuffled_kl_first = np.sum(kl_div(list(average_after_shuffle.iloc[0]), list(exp_training_last_trial.iloc[0])))
        shuffled_kl_last = np.sum(kl_div(list(average_after_shuffle.iloc[1]), list(exp_training_last_trial.iloc[0])))

        pD.append(shuffled_kl_first - shuffled_kl_last)

    # this sign makes sense because gT = kl_transfer_training - kl_pretraining_training
    p_val = len(np.where(pD >= gT)[0]) / (p * len(exp_training_last_trial.columns))
    print("p value of permutation test KL3", p_val)
    return None


def kl4(pretraining_control_prop, pretraining_exp_prop, transfer_control_prop, transfer_exp_prop,
        pretraining_control_count, pretraining_exp_count, transfer_control_count, transfer_exp_count):
    # KL[pre-training exp; first transfer trial exp] > KL[pre-training control; first transfer trial control]
    # experimental pretraining, control pretraining, transfer experimental, transfer control; no training

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

    gT = kl_exp - kl_control  # this should be larger, values further from 0?

    # filter pid so that exp pretraining = exp transfer and control pretraining = control transfer
    pretraining_exp_count = pretraining_exp_count[pretraining_exp_count.index.isin(transfer_exp_count.index)]
    pretraining_control_count = pretraining_control_count[
        pretraining_control_count.index.isin(transfer_control_count.index)]

    # create dict with actual counts
    strategy_dict = {}
    strategy_dict["exp_pretraining"] = pretraining_exp_count.values
    strategy_dict["control_pretraining"] = pretraining_control_count.values
    strategy_dict["exp_transfer"] = transfer_exp_count.iloc[:, 0].values  # get first column
    strategy_dict["control_transfer"] = transfer_control_count.iloc[:, 0].values  # get first column

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
        shuffled_kl_exp = np.sum(kl_div(average_after_shuffle.iloc[0].values, transfer_exp_prop.iloc[0].values))
        shuffled_kl_control = np.sum(kl_div(average_after_shuffle.iloc[1].values, transfer_control_prop.iloc[0].values))

        pD.append(shuffled_kl_exp - shuffled_kl_control)

    # gT = kl_exp - kl_control, null hypothesis needs to be other way round to reject it
    p_val = len(np.where(pD >= gT)[0]) / (
            p * len(
        pretraining_control_prop.columns))  # assuming number of participants  in control and exp are the same
    print("p value of permutation test KL4", p_val)
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
    ##### CREATE DATA
    pre_training = False
    filters = True
    # load large mouselab strategy data
    exp_large_mouselab_strategies = pd.read_pickle(
        "results/cm/inferred_strategies/exp_mouselab_test/strategies.pkl")
    control_large_mouselab_strategies = pd.read_pickle(
        "results/cm/inferred_strategies/control_mouselab_test/strategies.pkl")
    if pre_training:
        exp_pretraining_large_mouselab_test_strategies = pd.read_pickle(
            "results/cm/inferred_strategies/exp_mouselab_pretraining/strategies.pkl")
        control_pretraining_large_mouselab_test_strategies = pd.read_pickle(
            "results/cm/inferred_strategies/control_mouselab_pretraining/strategies.pkl")

    # load training small mouselab strategy data
    exp_training_strategies = pd.read_pickle("results/cm/inferred_strategies/exp_mouselab_training/strategies.pkl")

    # load score data
    mouselab_exp_score_data = pd.read_csv('data/human/exp_mouselab/mouselab-mdp.csv')
    mouselab_control_score_data = pd.read_csv('data/human/control_mouselab/mouselab-mdp.csv')

    # create cluster data, removes pid who did not change their strategy
    cluster_control_proportion_all, actual_count_control_transfer, control_good_participants_list, non_grouped_control = create_strategy_score_df(
        control_large_mouselab_strategies,
        mouselab_control_score_data, type="cluster",
        control_condition=True)
    cluster_exp_proportion_all, actual_count_exp_transfer, exp_good_participants_list, non_grouped_exp = create_strategy_score_df(
        exp_large_mouselab_strategies,
        mouselab_exp_score_data, type="cluster",
        control_condition=False)
    if pre_training:
        cluster_control_pretraining_proportion_all, actual_count_control_pretraining_transfer, pretraining_control_good_participants_list, non_grouped_pretraining_control = create_strategy_score_df(
            control_pretraining_large_mouselab_test_strategies,
            mouselab_control_score_data, type="cluster",
            control_condition=True)
        cluster_exp_pretraining_proportion_all, actual_count_exp_pretraining_transfer, pretraining_exp_good_participants_list, non_grouped_pretraining_exp = create_strategy_score_df(
            exp_pretraining_large_mouselab_test_strategies,
            mouselab_control_score_data, type="cluster",
            control_condition=False)

    ### create strategy data, needed for score, filter by good participants
    control_proportion_all = create_strategy_score_df(control_large_mouselab_strategies,
                                                      mouselab_control_score_data, type="strategy",
                                                      control_condition=True)
    filtered_exp_large_mouselab_test_strategies = {your_key: exp_large_mouselab_strategies[your_key] for your_key
                                                   in exp_good_participants_list}
    exp_proportion_all = create_strategy_score_df(
        filtered_exp_large_mouselab_test_strategies,
        mouselab_exp_score_data, type="strategy",
        control_condition=False)
    # create_strategy_proportion_plots(control_proportion_all, exp_proportion_all, proportion_adaptive, proportion_maladaptive)

    ##### ANALYSIS 1
    training_exp, transfer_exp, transfer_proportion, actual_number_of_transfer, total_number_of_filtered_participants = check_for_same_strategy_before_after_transfer(
        exp_large_mouselab_strategies, exp_training_strategies, filters)

    ### chi-squared test for transfer proportion
    # The chi-square test tests the null hypothesis that the categorical data has the given frequencies.
    print("Chi-squared test to test for transfer proportion: ",
          {chisquare(
              [(actual_number_of_transfer), ((total_number_of_filtered_participants - actual_number_of_transfer))],
              f_exp=[(1 / 7) * total_number_of_filtered_participants,
                     (6 / 7) * total_number_of_filtered_participants])})
    #
    # value1 = [actual_number_of_transfer, total_number_of_filtered_participants - actual_number_of_transfer]
    # value2 = [0.2 * total_number_of_filtered_participants, 0.8 * total_number_of_filtered_participants]
    # m = np.row_stack([value1, value2])
    # res = stats.fisher_test(m, simulate_p_value=True)
    # print(f"Fisher exact test to test for transfer proportion: p={res[0][0]:.3f}, odds={res[1][0]}")

    ##### ANALYSIS 2
    ### plots
    # create_cluster_proportion_plots(cluster_control_proportion_all, cluster_exp_proportion_all, cluster_proportion_adaptive, cluster_proportion_maladaptive)
    # if pre_training is False:
    #     cluster_control_pretraining_proportion_all, cluster_exp_pretraining_proportion_all = None, None
    # create_plots_training_test(training_exp, cluster_control_proportion_all, cluster_exp_proportion_all, pre_training, cluster_control_pretraining_proportion_all, cluster_exp_pretraining_proportion_all)
    # plot_strategy_distribution(training_exp, cluster_control_proportion_all, cluster_exp_proportion_all, pre_training,
    #                            cluster_control_pretraining_proportion_all, cluster_exp_pretraining_proportion_all)
    # plot_cluster_proportions(training_exp, transfer_exp)

    ### KL test #todo: currently this KL analysis use the filter criterion from ANALYSIS 1
    # kl1(training_exp, transfer_exp, cluster_exp_proportion_all)
    # kl2(training_exp, transfer_exp, cluster_exp_proportion_all,  actual_count_control_transfer, cluster_control_proportion_all)
    # kl3(training_exp, cluster_exp_proportion_all, cluster_exp_pretraining_proportion_all,
    #     non_grouped_pretraining_exp, transfer_exp)
    # if pre_training:
    #     kl4(cluster_control_pretraining_proportion_all, cluster_exp_pretraining_proportion_all,
    #         cluster_control_proportion_all, cluster_exp_proportion_all,
    #         non_grouped_pretraining_control, non_grouped_pretraining_exp, non_grouped_control, non_grouped_exp)

    ### Fisher test to test between two conditions of first trial of control and experimental
    value1 = actual_count_control_transfer.iloc[0].values
    # value1 = np.pad(value1, (0, 1), 'constant')
    value2 = actual_count_exp_transfer.iloc[0].values
    m = np.row_stack([value1, value2])
    m = m.astype(np.int64)
    m_prop = m / 36
    res = stats.fisher_test(m, simulate_p_value=True)
    print(f"Fisher exact test of distribution of first trial between control and experimental: p={res[0][0]:.3f}")
    #
    #
    # if pre_training:
    #     ## Fisher test to test between pretraining and first transfer in experimental condition
    #     # add the missing columsn
    #     if actual_count_exp_pretraining_transfer.shape[1] != actual_count_exp_transfer.shape[1]:
    #         missing_columns = actual_count_exp_transfer.columns.difference(
    #             actual_count_exp_pretraining_transfer.columns).tolist()
    #         for column in missing_columns:
    #             actual_count_exp_pretraining_transfer[column] = 0
    #     ### experimental condition
    #     # reindex, i.e. reorder the columns
    #     actual_count_exp_pretraining_transfer = actual_count_exp_pretraining_transfer.reindex(
    #         sorted(actual_count_exp_pretraining_transfer.columns), axis=1)
    #     actual_count_exp_transfer = actual_count_exp_transfer.reindex(sorted(actual_count_exp_transfer.columns), axis=1)
    #
    #     value1 = actual_count_exp_pretraining_transfer.iloc[0].values.astype(float)
    #     value2 = actual_count_exp_transfer.iloc[0].values.astype(float)
    #     m = np.row_stack([value1, value2])
    #     res = stats.fisher_test(m, simulate_p_value=True)
    #     print(
    #         f"Fisher exact test of distribution of pretraining and transfer in experimental condition: p={res[0][0]:.3f}, odds={res[1][0]}")
    #
    # # ### The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution.
    print("Mean score control", control_proportion_all["score"].mean())
    print("Mean score exp", exp_proportion_all["score"].mean())
    print("STD score control", control_proportion_all["score"].std())
    print("STD score exp", exp_proportion_all["score"].std())
    # # print("Wilcoxon rank sum test of score between control and experimental: ",
    # #       ranksums(control_proportion_all["score"][:5], exp_proportion_all["score"][:5], alternative="less"))
    print("Wilcoxon rank sum test of score between control and experimental: ",
          ranksums(exp_proportion_all["score"], control_proportion_all["score"], alternative="greater"))
    #
    # ### trend test
    # print("Trend test for the control group")
    # trend_tests(cluster_control_proportion_all)
    # print("Trend test for the experimental group")
    # trend_tests(cluster_exp_proportion_all)
    #
    ### Fisher test between first training and last training
    exp_training_grouped = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                 "Middle planning", "Frugal planning", "Other planning"], index=[0, 1])
    for columns in exp_training_grouped:
        exp_training_grouped[columns].iloc[0] = list(training_exp.iloc[1]).count(columns)
        exp_training_grouped[columns].iloc[1] = list(training_exp.iloc[-1]).count(columns)
    value1 = exp_training_grouped.iloc[0].values.astype(float)
    value2 = exp_training_grouped.iloc[1].values.astype(float)
    m = np.row_stack([value1, value2])
    res = stats.fisher_test(m, simulate_p_value=True)
    print(f"Fisher exact test of distribution of first training and last training: p={res[0][0]:.3f}, odds={res[1][0]}")

    if pre_training:
        control_data = anvoca_preprocessing(input_dataframe=mouselab_control_score_data, condition="control")
        exp_data = anvoca_preprocessing(input_dataframe=mouselab_exp_score_data, condition="exp")

        df_final = pd.concat([control_data, exp_data])
        df_final.to_csv(f"mouselab_anvoca.csv")
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
