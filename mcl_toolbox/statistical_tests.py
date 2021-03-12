import sys
import pandas as pd
import numpy as np
import random
import pymannkendall as mk
from scipy.stats import friedmanchisquare
from scipy.stats import mannwhitneyu

from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.utils.statistics_utils import create_comparable_data

"""
This script runs statistical tests that tests whether:
1. strategy development and overall strategy frequency is significantly different across conditions
2. cluster development and overall cluster frequency is significantly different across conditions
3. decision system development and overall decision system frequency is significantly different across conditions

A mannwhitneyu test will be used to test whether the distributions of two independent samples are equal or not.
A friedmanchisquare test will be used to whether the distributions of two or more paired samples are equal or not.
A Mann Kendall test is used to test for trends

"""

random.seed(123)
reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}


def load_data_from_computational_microscope(exp, exp_num, reward_exps):
    block = "training"
    strategy_space = learning_utils.pickle_load("data/strategy_space.pkl")
    features = learning_utils.pickle_load("data/microscope_features.pkl")
    strategy_weights = learning_utils.pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")
    features = learning_utils.pickle_load("data/microscope_features.pkl")
    decision_systems = learning_utils.pickle_load("data/decision_systems.pkl")
    feature_systems = learning_utils.pickle_load("data/feature_systems.pkl")
    decision_system_features = learning_utils.pickle_load("data/decision_system_features.pkl")
    DS_proportions = learning_utils.pickle_load("data/strategy_decision_proportions.pkl")
    W_DS = learning_utils.pickle_load("data/strategy_decision_weights.pkl")
    cluster_map = learning_utils.pickle_load("data/kl_cluster_map.pkl")
    strategy_scores = learning_utils.pickle_load("data/strategy_scores.pkl")
    cluster_scores = learning_utils.pickle_load("data/cluster_scores.pkl")

    if exp_num == "c2.1":
        pipeline = exp_pipelines["c2.1_dec"]
    else:
        pipeline = exp_pipelines[exp_num]
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = learning_utils.get_normalized_features(reward_exps[exp])
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)

    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)
    pids = None
    exp = Experiment(exp_num=exp_num, cm=cm, pids=pids, block=block)

    if exp_num == "c2.1":
        dir_path = f"../results/inferred_strategies/c2.1_dec"
    else:
        dir_path = f"../results/inferred_strategies/{exp_num}"
    if block:
        dir_path += f"_{block}"

    try:
        strategies = learning_utils.pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = learning_utils.pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print("Exception", e)

    strategy_proportions, strategy_proportions_trialwise, cluster_proportions, cluster_proportions_trialwise, decision_system_proportions, mean_dsw = exp.statistical_kpis(
        features,
        normalized_features,
        strategy_weights,
        decision_systems,
        W_DS, DS_proportions,
        strategy_scores,
        cluster_scores,
        cluster_map,
        precomputed_strategies=strategies,
        precomputed_temperatures=temperatures,
        show_pids=False)
    return strategy_proportions, strategy_proportions_trialwise, cluster_proportions, cluster_proportions_trialwise, decision_system_proportions, mean_dsw

def test_for_equal_distribution(name_distribution_dict: dict, type: str):
    """
    Are the distributions of the proportions between the environments equal?
    friedmanchisquare for all and mannwhitneyu tests for pairs

    Args:
        name_distribution_dict: a dictionary with the variance type and corresponding distribution }
        e.g. {"increasing": [array]}

    Returns: None

    """
    length_of_dict = len(name_distribution_dict)
    print(f" === Test for Equal distributions between the {type} === ")

    if length_of_dict >= 3:
        stat, p = friedmanchisquare(*[v for k, v in name_distribution_dict.items()])
        print('Friedman chi-squared tests: stat=%.3f, p=%.3f' % (stat, p))

    for variance_type_a, distribution_a in name_distribution_dict.items():
        for variance_type_b, distribution_b in name_distribution_dict.items():
            stat, p = mannwhitneyu(distribution_a, distribution_b)
            print(f"{variance_type_a} vs {variance_type_b}:  stat={stat:.3f}, p={p:.3f}")

def create_data_for_distribution_test(strategy_name_dict: dict):
    """
    Create data to check for equal distribution
    Args:
        strategy_name_dict: same as reward_exps:
        reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}

    Returns:

    """
    # name: increasing
    # experiment v1.0
    column_names = ["increasing_variance", "decreasing_variance", "constant_variance"]
    cluster_df = pd.DataFrame(columns=column_names)
    strategy_df = pd.DataFrame(columns=column_names)
    decision_system_df = pd.DataFrame(columns=column_names)

    for name, experiment in strategy_name_dict.items(): #name: increasing/decreasing, experiment: v1.0
        strategy_proportions, _, cluster_proportions, _, decision_system_proportions, _ = load_data_from_computational_microscope(
            name, experiment, strategy_name_dict)
        strategy_df[name] = list(create_comparable_data(strategy_proportions, len=89).values())
        cluster_df[name] = list(create_comparable_data(cluster_proportions, len=14).values())
        decision_system_df[name] = decision_system_proportions["Relative Influence (%)"].tolist()
    return strategy_df, cluster_df, decision_system_df

def create_data_for_trend_test(strategy_name_dict: dict, trend_test: True):
    """
    Create data for the trend tests
    Args:
        strategy_name_dict: reward_exps
        reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}

    Returns: trend data as pandas dataframes
    #todo: decision trend, not really used but might be useful for completeness
    """
    column_names = ["increasing_variance", "decreasing_variance", "constant_variance"]
    cluster_trend = pd.DataFrame(columns=column_names)
    strategy_trend = pd.DataFrame(columns=column_names)
    # decision_trend = pd.DataFrame(columns=column_names)

    for name, experiment in strategy_name_dict.items():
        _, strategy_proportions_trialwise, _, cluster_proportions_trialwise, _, mean_dsw = load_data_from_computational_microscope(
            name, experiment, strategy_name_dict)

        strategy_temp = []
        cluster_temp = []
        ds_temp = []
        for i in range(0, len(strategy_proportions_trialwise)):
            strategy_temp.append(list(create_comparable_data(strategy_proportions_trialwise[i], len=89).values()))
        if trend_test:
            strategy_trend[name] = list(map(list, zip(*strategy_temp)))  # transpose
        else:
            strategy_trend[name] = strategy_temp

        for i in range(0, len(cluster_proportions_trialwise)):
            cluster_temp.append(list(create_comparable_data(cluster_proportions_trialwise[i], len=14).values()))
        if trend_test:
            cluster_trend[name] = list(map(list, zip(*cluster_temp)))
        else:
            cluster_trend[name] = cluster_temp

        # for i in range(0, len(mean_dsw)):
        #     ds_temp.append(list(create_comparable_data(mean_dsw[i], len=5).values()))
        # decision_trend[name] = ds_temp

    return strategy_trend, cluster_trend  # , decision_trend

def test_for_trend(trend, analysis_type: str):
    # analysis_type: strategy or cluster or ds
    for columns in trend:
        for strategy_number in range(0, len(trend["increasing_variance"])):
            test_results = mk.original_test(trend[columns][strategy_number])
            print(f"Mann Kendall Test: {columns} {analysis_type}: ", strategy_number, test_results)

def test_first_trials_vs_last_trials(trend, n, analysis_type):
    """
    This function tests whether the distributions between the first n trials and the last n trials are equal.
    Args:
        trend: pandas dataframe with the variance types as header and number of trials as rows
        n: number of first and last trials to take into consideration
        analysis_type: strategy or strategy cluster

    Returns:

    """
    # todo: add decision systems

    average_first_n_trials = trend.iloc[0:n].sum()  # add first n rows
    average_last_n_trials = trend.iloc[-(n + 1):-1, :].sum()  # add last n rows
    for columns in trend:
        stat, p = mannwhitneyu(average_first_n_trials[columns], average_last_n_trials[columns])
        print(f'{analysis_type}, {columns}: First{n} trials vs Last {n} trials: stat=%.3f, p=%.3f' % (stat, p))


print(" ----------------- Distribution Difference -----------------")
strategy_df, cluster_df, decision_system_df = create_data_for_distribution_test(reward_exps)

strategy_difference_dict = {"increasing": strategy_df["increasing_variance"],
                            "decreasing": strategy_df["decreasing_variance"],
                            "constant": strategy_df["constant_variance"]}
test_for_equal_distribution(strategy_difference_dict, "Strategies")

cluster_difference_dict = {"increasing": cluster_df["increasing_variance"],
                           "decreasing": cluster_df["decreasing_variance"],
                           "constant": cluster_df["constant_variance"]}
test_for_equal_distribution(cluster_difference_dict, "Strategy Clusters")

decision_system_difference_dict = {"increasing": decision_system_df["increasing_variance"],
                                   "decreasing": decision_system_df["decreasing_variance"],
                                   "constant": decision_system_df["constant_variance"]}
test_for_equal_distribution(decision_system_difference_dict, "Decision Systems")

print(" ----------------- Trends -----------------")
strategy_trend, cluster_trend = create_data_for_trend_test(reward_exps, trend_test=True)
test_for_trend(strategy_trend, "Strategy")
test_for_trend(cluster_trend, "Strategy Cluster")
# test_for_trend(decision_trend, "Decision System")

# print(" ----------------- Decision System -----------------")
# for i in range(0, 5):
#     increasing_ds_trend = mk.original_test(list(mean_dsw_increasing[:, i]))
#     print("Mann Kendall Test: Increasing Decision System: ", i, increasing_ds_trend)
#
# for i in range(0, 5):
#     decreasing_ds_trend = mk.original_test(list(mean_dsw_decreasing[:, i]))
#     print("Mann Kendall Test: Decreasing Decision System: ", i, decreasing_ds_trend)
#
# for i in range(0, 5):
#     constant_ds_trend = mk.original_test(list(mean_dsw_constant[:, i]))
#     print("Mann Kendall Test: Constant Decision System: ", i, constant_ds_trend)

print(" ----------------- First vs Last trial -----------------")
first_last_strategies, first_last_clusters = create_data_for_trend_test(reward_exps, trend_test=False)
test_first_trials_vs_last_trials(first_last_strategies, 2, "Strategy")
test_first_trials_vs_last_trials(first_last_clusters, 2, "Strategy Cluster")
# get_first_n_trials(decision_trend, 2, "Decision System")
