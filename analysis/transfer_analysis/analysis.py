import numpy as np
import pandas as pd
import os
import ast
from sklearn.cluster import KMeans
import collections
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
stats = importr("stats")

"""
Files that contains analysis that applies to all conditions
i.e. was there learning? 
"""
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


def create_cluster_df(data, pretraining=False, transfer=True):
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
                       "Other planning": [5, 13]}

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
                                                                                               transfer=False)
        return cluster_proportion_all, actual_count, good_participants_list, data

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


def check_for_same_strategy_before_after_transfer(experiment_type, experiment, transferdata):
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
    last_n_entries = clusters_mapped.tail(3)
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

def check_for_same_strategy_before_after_transfer_mouselab(exp_large_mouselab_test_strategies, exp_training_strategies):
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
    pid_level_data_transfer, _, actual_count_test = create_cluster_df(exp_large_mouselab_test_strategies, transfer=True)
    # get last strategy in training
    pid_level_data_training, _, _, actual_count_training = create_cluster_df(exp_training_strategies, transfer=False)

    # get intersection of pid between training and transfer data (some training pid were removed)
    pid_list = list(set(pid_level_data_transfer.index) & set(pid_level_data_training.index))
    pid_level_data_transfer = pid_level_data_transfer[pid_level_data_transfer.index.isin(pid_list)]

    # filter out participants who did not change their strategy in the last 3 trials of training
    good_participants_list = []
    for index, row in pid_level_data_training.iterrows():
        if len(row[-3:].unique()) == 1:
            good_participants_list.append(index)

    pid_level_data_transfer = pid_level_data_transfer[pid_level_data_transfer.index.isin(good_participants_list)]
    pid_level_data_training = pid_level_data_training[pid_level_data_training.index.isin(good_participants_list)]

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
    df_assigned["trial"] = list(range(0, number_of_test_trials)) * number_of_pid #todo: add pretraining as label in the data, otherwise all transfer trials are labelled as "test"
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

def filter_pid_who_changed_strategy_during_training(data):
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
    filtered_data = data
    return filtered_data

if __name__ == "__main__":
    pre_training = False
    ##### CREATE DATA FOR MOUSELAB

    # load large mouselab strategy data
    exp_large_mouselab_strategies = pd.read_pickle(
        "results/cm/inferred_strategies/exp_mouselab_test/strategies.pkl")
    control_large_mouselab_strategies = pd.read_pickle(
        "results/cm/inferred_strategies/control_mouselab_test/strategies.pkl")
    if pre_training:
        exp_pretraining_large_mouselab_test_strategies = pd.read_pickle(
            "results/cm/inferred_strategies/exp_mouselab_pre-training/strategies.pkl")
        control_pretraining_large_mouselab_test_strategies = pd.read_pickle(
            "results/cm/inferred_strategies/control_mouselab_pre-training/strategies.pkl")

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

    training_exp_mouselab, transfer_exp, transfer_proportion, actual_number_of_transfer, total_number_of_filtered_participants = check_for_same_strategy_before_after_transfer_mouselab(
        exp_large_mouselab_strategies, exp_training_strategies)




    ### ROADTRIP DATA
    number_of_training_trials = 10
    number_of_test_trials = 15

    ### load data
    exp_data_roadtrip = pd.read_csv('data/human/exp_roadtrip/mouselab-mdp.csv')
    control_data_roadtrip = pd.read_csv('data/human/control_roadtrip/mouselab-mdp.csv')
    experiment = "roadtrip"

    ##### CREATE DATA
    non_grouped_control_df, grouped_control_df, actual_count_control_transfer = create_trial_df_for_each_participant_group(
        control_data_roadtrip, experiment, pre_training=False)
    non_grouped_exp_df, grouped_exp_df, actual_count_exp_transfer = create_trial_df_for_each_participant_group(
        exp_data_roadtrip, experiment, pre_training=False)
    if pre_training:
        non_grouped_control_pretraining_df, grouped_control_pretraining_df, actual_count_control_pretraining_transfer = create_trial_df_for_each_participant_group(
            control_data_roadtrip, experiment, pre_training=True)
        non_grouped_exp_pretraining_df, grouped_exp_pretraining_df, actual_count_exp_pretraining_transfer = create_trial_df_for_each_participant_group(
            exp_data_roadtrip, experiment, pre_training=True)

    ### filter non_grouped_df for participants who changed their strategy during training (exp only)
    filtered_non_grouped_exp_df = filter_pid_who_changed_strategy_during_training(non_grouped_exp_df)

    ##### ANALYSIS 1
    ### find out how many people used the same strategy type at the end of training and beginning of transfer, only for exp condition
    training_exp_roadtrip, transfer_exp, transfer_proportion, actual_number_of_transfer, total_number_of_filtered_participants = check_for_same_strategy_before_after_transfer(
        "exp", experiment=experiment, transferdata=filtered_non_grouped_exp_df)



    ### MORTGAGE DATA
    exp_data_mortgage = pd.read_csv('data/human/exp_mortgage/mouselab-mdp.csv')
    control_data_mortgage = pd.read_csv('data/human/control_mortgage/mouselab-mdp.csv')
    experiment = "mortgage"

    ##### CREATE DATA
    non_grouped_control_df, grouped_control_df, actual_count_control_transfer = create_trial_df_for_each_participant_group(
        control_data_mortgage, experiment, pre_training=False)
    non_grouped_exp_df, grouped_exp_df, actual_count_exp_transfer = create_trial_df_for_each_participant_group(
        exp_data_mortgage, experiment, pre_training=False)
    if pre_training:
        non_grouped_control_pretraining_df, grouped_control_pretraining_df, actual_count_control_pretraining_transfer = create_trial_df_for_each_participant_group(
            control_data, experiment, pre_training=True)
        non_grouped_exp_pretraining_df, grouped_exp_pretraining_df, actual_count_exp_pretraining_transfer = create_trial_df_for_each_participant_group(
            exp_data, experiment, pre_training=True)

    ### filter non_grouped_df for participants who changed their strategy during training (exp only)
    filtered_non_grouped_exp_df = filter_pid_who_changed_strategy_during_training(non_grouped_exp_df)

    ##### ANALYSIS 1
    ### find out how many people used the same strategy type at the end of training and beginning of transfer, only for exp condition
    training_exp_mortgage, transfer_exp, transfer_proportion, actual_number_of_transfer, total_number_of_filtered_participants = check_for_same_strategy_before_after_transfer(
        "exp", experiment=experiment, transferdata=filtered_non_grouped_exp_df)

    ### Fisher test between first training and last training
    exp_training_grouped = pd.DataFrame(columns=["Goal setting", "Backward planning", "Forward planning",
                                                 "Middle planning", "Frugal planning", "Other planning"], index=[0, 1])
    # print("mouselab", training_exp_mouselab.T.iloc[1].shape)
    # print("roadtrip", training_exp_roadtrip.iloc[1].shape)
    # print("mortgage", training_exp_mortgage.iloc[1].shape)
    for columns in exp_training_grouped:
        exp_training_grouped[columns].iloc[0] = list(training_exp_mouselab.T.iloc[1]).count(columns) + list(training_exp_roadtrip.iloc[1]).count(columns) + list(training_exp_mortgage.iloc[1]).count(columns)
        exp_training_grouped[columns].iloc[1] = list(training_exp_mouselab.T.iloc[-1]).count(columns) + list(training_exp_roadtrip.iloc[-1]).count(columns) + list(training_exp_mortgage.iloc[-1]).count(columns)
    value1 = exp_training_grouped.iloc[0].values.astype(float)
    value2 = exp_training_grouped.iloc[1].values.astype(float)
    m = np.row_stack([value1, value2])
    res = stats.fisher_test(m, simulate_p_value=True)
    print(f"Fisher exact test of distribution of first training and last training WITHOUT FILTER: p={res[0][0]:.3f}")