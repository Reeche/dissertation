import sys
import pandas as pd
import numpy as np
import random
import itertools
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
import pymannkendall as mk
from utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

from utils.experiment_utils import Experiment
from computational_microscope.computational_microscope import ComputationalMicroscope
from utils.statistics_utils import create_comparable_data

"""
This script runs statistical tests that tests whether:
1. strategy development and overall strategy frequency is significantly different across conditions
2. cluster development and overall cluster frequency is significantly different across conditions
3. decision system development and overall decision system frequency is significantly different across conditions

A mannwhitneyu test will be used to test whether the distributions of two independent samples are equal or not.
A friedmanchisquare test will be used to whether the distributions of two or more paired samples are equal or not.
A Mann Kendall test is used to test for trends

"""
# todo: this code seriously need some tidy up

def load_data_from_computational_microscope(keys, values):
    random.seed(123)

    block = "training"

    reward_exps = {"increasing_variance": "v1.0",
                   "decreasing_variance": "c2.1",
                   "constant_variance": "c1.1"}
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

    if values == "c2.1":
        pipeline = exp_pipelines["c2.1_dec"]
    else:
        pipeline = exp_pipelines[values]
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = learning_utils.get_normalized_features(reward_exps[keys])
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)

    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)
    pids = None
    exp = Experiment(exp_num=values, cm=cm, pids=pids, block=block)

    dir_path = f"../results/inferred_strategies/{keys}"
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


# todo: change this so it can take a variable amount of args
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



# create data
strategy_proportions_increasing, strategy_proportions_trialwise_increasing, cluster_proportions_increasing, cluster_proportions_trialwise_increasing, decision_system_proportions_increasing, mean_dsw_increasing = load_data_from_computational_microscope(
    "increasing_variance", "v1.0")
strategy_proportions_constant, strategy_proportions_trialwise_constant, cluster_proportions_constant, cluster_proportions_trialwise_constant, decision_system_proportions_constant, mean_dsw_constant = load_data_from_computational_microscope(
    "constant_variance", "c1.1")
strategy_proportions_decreasing, strategy_proportions_trialwise_decreasing, cluster_proportions_decreasing, cluster_proportions_trialwise_decreasing, decision_system_proportions_decreasing, mean_dsw_decreasing = load_data_from_computational_microscope(
    "decreasing_variance", "c2.1")

# create the data for clusters, need to make them all equal length
# len 14 because there are 13 strategy clusters
increasing_cluster = create_comparable_data(cluster_proportions_increasing, len=14)
decreasing_cluster = create_comparable_data(cluster_proportions_decreasing, len=14)
constant_cluster = create_comparable_data(cluster_proportions_constant, len=14)

increasing_ds = decision_system_proportions_increasing["Relative Influence (%)"].tolist()
decreasing_ds = decision_system_proportions_decreasing["Relative Influence (%)"].tolist()
constant_ds = decision_system_proportions_constant["Relative Influence (%)"].tolist()

# create the data for strategies, need to make them all equal length
# len 89 because there are 89 strategies strategy clusters
increasing_strategy = create_comparable_data(strategy_proportions_increasing, len=89)
decreasing_strategy = create_comparable_data(strategy_proportions_decreasing, len=89)
constant_strategy = create_comparable_data(strategy_proportions_constant, len=89)


print(" ----------------- Distribution Difference -----------------")
strategy_difference_dict = {"increasing": list(increasing_strategy.values()), "decreasing": list(decreasing_strategy.values()),
        "constant": list(constant_strategy.values())}
test_for_equal_distribution(strategy_difference_dict, "Strategies")

cluster_difference_dict = {"increasing": list(increasing_cluster.values()), "decreasing": list(decreasing_cluster.values()),
                           "constant": list(constant_cluster.values())}
test_for_equal_distribution(cluster_difference_dict, "Strategy Clusters")

decision_system_difference_dict = {"increasing": increasing_ds, "decreasing": decreasing_ds,
                           "constant": constant_ds}
test_for_equal_distribution(decision_system_difference_dict, "Decision Systems")



### Seasonality
print(" ----------------- Trends -----------------")

# print(" ----------------- Strategies -----------------")
# # Strategies
# strategy_list_increasing = []
# strategy_list_decreasing = []
# strategy_list_constant = []
# for i in range(0, len(strategy_proportions_trialwise_increasing)):
#     strategy_list_increasing.append(
#         list(create_comparable_data(strategy_proportions_trialwise_increasing[i], len=79).values()))
#     strategy_list_decreasing.append(
#         list(create_comparable_data(strategy_proportions_trialwise_decreasing[i], len=79).values()))
#     strategy_list_constant.append(
#         list(create_comparable_data(strategy_proportions_trialwise_constant[i], len=79).values()))


print(" ----------------- First vs. last trial -----------------")
print(" ----------------- Strategies distribution -----------------")

#
# strategy_array_increasing = np.array(strategy_list_increasing)
# strategy_array_decreasing = np.array(strategy_list_decreasing)
# strategy_array_constant = np.array(strategy_list_constant)
#
# for i in range(0, 79):
#     increasing_strategy_trend = mk.original_test(list(strategy_array_increasing[:, i]))
#     print("Mann Kendall Test: Increasing Strategies: ", i, increasing_strategy_trend)
#
# for i in range(0, 79):
#     decreasing_strategy_trend = mk.original_test(list(strategy_array_decreasing[:, i]))
#     print("Mann Kendall Test: decreasing Strategies: ", i, decreasing_strategy_trend)
#
# for i in range(0, 79):
#     constant_strategy_trend = mk.original_test(list(strategy_array_constant[:, i]))
#     print("Mann Kendall Test: Constant Strategies: ", i, constant_strategy_trend)

print(" ----------------- Clusters -----------------")
cluster_mapping = ["Goal-setting with exhaustive backward planning",
                   "Forward planning strategies similar to Breadth First Search",
                   "Middle-out planning",
                   "Forward planning strategies similar to Best First Search",
                   "Local search",
                   "Maximizing Goal-setting with exhaustive backward planning",
                   "Frugal planning",
                   "Myopic planning",
                   "Maximizing goal-setting with limited backward planning",
                   "Frugal goal-setting strategies",
                   "Strategy that explores immediate outcomes on the paths to the best final outcomes",
                   "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing",
                   "Miscellaneous strategies"]

cluster_list_increasing = []
cluster_list_decreasing = []
cluster_list_constant = []
for i in range(0, len(cluster_proportions_trialwise_increasing)):
    cluster_list_increasing.append(
        list(create_comparable_data(cluster_proportions_trialwise_increasing[i], len=14).values()))
    cluster_list_decreasing.append(
        list(create_comparable_data(cluster_proportions_trialwise_decreasing[i], len=14).values()))
    cluster_list_constant.append(
        list(create_comparable_data(cluster_proportions_trialwise_constant[i], len=14).values()))

cluster_array_increasing = np.array(cluster_list_increasing)
cluster_array_decreasing = np.array(cluster_list_decreasing)
cluster_array_constant = np.array(cluster_list_constant)

for i in range(0, 14):
    increasing_cluster_trend = mk.original_test(list(cluster_array_increasing[:, i]))
    print("Mann Kendall Test: Increasing Cluster: ", i, increasing_cluster_trend)

for i in range(0, 14):
    decreasing_cluster_trend = mk.original_test(list(cluster_array_decreasing[:, i]))
    print("Mann Kendall Test: Decreasing Cluster: ", i, decreasing_cluster_trend)

for i in range(0, 14):
    constant_cluster_trend = mk.original_test(list(cluster_array_constant[:, i]))
    print("Mann Kendall Test: Constant Cluster: ", i, constant_cluster_trend)

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


# print(" ----------------- First vs. last trial -----------------")
# print(" ----------------- Strategies -----------------")
# n = 2
#
# average_first_n_trials = np.mean(strategy_array_increasing[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(strategy_array_increasing[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Increasing: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
#
# average_first_n_trials = np.mean(strategy_array_decreasing[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(strategy_array_decreasing[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Decreasing: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
#
# average_first_n_trials = np.mean(strategy_array_constant[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(strategy_array_constant[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Constant: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
#
# print(" ----------------- Strategy cluster  -----------------")
#
# average_first_n_trials = np.mean(cluster_array_increasing[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(cluster_array_increasing[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Increasing: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
#
# average_first_n_trials = np.mean(cluster_array_decreasing[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(cluster_array_decreasing[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Decreasing: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
#
# average_first_n_trials = np.mean(cluster_array_constant[:n,:], axis=0) #first 5 elements
# average_last_n_trials = np.mean(cluster_array_constant[-(n+1):-1,:], axis=0) #last 5 elements
# stat, p = mannwhitneyu(average_first_n_trials, average_last_n_trials)
# print('Constant: First %.0f trials vs Last %.0f trials: stat=%.3f, p=%.3f' % (n, n, stat, p))
