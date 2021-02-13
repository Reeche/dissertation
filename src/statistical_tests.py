import sys
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
import numpy as np
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

"""

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

block = "training"

reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}


# need to get the strategy / cluster / decision system proportions from each condition

def get_data_of_cluster_decision_system(keys, values):
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

    strategy_proportions, cluster_proportions, decision_system_proportions = exp.statistical_kpis(features,
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
    return strategy_proportions, cluster_proportions, decision_system_proportions



# for keys, values in reward_exps.items():
strategy_proportions_increasing, cluster_proportions_increasing, decision_system_proportions_increasing = get_data_of_cluster_decision_system(
    "increasing_variance", "v1.0")
strategy_proportions_constant, cluster_proportions_constant, decision_system_proportions_constant = get_data_of_cluster_decision_system(
    "constant_variance", "c1.1")
strategy_proportions_decreasing, cluster_proportions_decreasing, decision_system_proportions_decreasing = get_data_of_cluster_decision_system(
    "decreasing_variance", "c2.1")

increasing = create_comparable_data(cluster_proportions_increasing, len=14)
decreasing = create_comparable_data(cluster_proportions_decreasing, len=14)
constant = create_comparable_data(cluster_proportions_constant, len=14)

print(" ----------------- Clusters -----------------")
print(np.sum(list(increasing.values())))
print(np.sum(list(decreasing.values())))
print(np.sum(list(constant.values())))
stat, p = friedmanchisquare(list(increasing.values()), list(decreasing.values()), list(constant.values()))
print('Friedman chi-squared tests: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(list(increasing.values()), list(decreasing.values()))
print('Increasing vs Decreasing: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(list(increasing.values()), list(constant.values()))
print('Increasing vs Constant: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(list(decreasing.values()), list(constant.values()))
print('Decreasing vs Constant: stat=%.3f, p=%.3f' % (stat, p))



print(" ----------------- Decision systems -----------------")
print(np.sum(decision_system_proportions_increasing["Relative Influence (%)"].tolist()))
print(np.sum(decision_system_proportions_decreasing["Relative Influence (%)"].tolist()))
print(np.sum(decision_system_proportions_constant["Relative Influence (%)"].tolist()))
stat, p = friedmanchisquare(decision_system_proportions_increasing["Relative Influence (%)"].tolist(),
                            decision_system_proportions_decreasing["Relative Influence (%)"].tolist(),
                            decision_system_proportions_constant["Relative Influence (%)"].tolist())
print('Friedman chi-squared tests: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(decision_system_proportions_increasing["Relative Influence (%)"].tolist(), decision_system_proportions_decreasing["Relative Influence (%)"].tolist())
print('Increasing vs Decreasing: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(decision_system_proportions_increasing["Relative Influence (%)"].tolist(), decision_system_proportions_constant["Relative Influence (%)"].tolist())
print('Increasing vs Constant: stat=%.3f, p=%.3f' % (stat, p))

stat, p = mannwhitneyu(decision_system_proportions_decreasing["Relative Influence (%)"].tolist(), decision_system_proportions_constant["Relative Influence (%)"].tolist())
print('Decreasing vs Constant: stat=%.3f, p=%.3f' % (stat, p))

#todo: statistical tests comparing the beginning of the trial and end of the trial: decision system and cluster proportions
