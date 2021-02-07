import sys
from utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
from utils.experiment_utils import Experiment
from computational_microscope.computational_microscope import ComputationalMicroscope

"""
This script runs statistical tests that tests whether:
1. strategy development and overall strategy frequency is significantly different across conditions
2. cluster development and overall cluster frequency is significantly different across conditions
3. decision system development and overall decision system frequency is significantly different across conditions

A 2 independent sample t-test will be used

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

#exp_num = ["v1.0", "c1.1", "c2.1"]
block = "training"

reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1",
               "transfer_task": "T1.1"}

# need to get the strategy / cluster / decision system proportions from each condition
for keys, values in reward_exps.items():
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

    # if values == "c2.1_dec":
    #     exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    # else:
    #     exp = Experiment(values, cm=cm, pids=pids, block=block)

    dir_path = f"../results/inferred_strategies/{keys}"
    if block:
        dir_path += f"_{block}"

    try:
        strategies = learning_utils.pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = learning_utils.pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print("Exception", e)

    strategy_proportions, cluster_proportions, decision_system_proportions = exp.statistical_kpis(features, normalized_features, strategy_weights,
                                                                                                  decision_systems, W_DS, DS_proportions, strategy_scores,
                                                                                                  cluster_scores, cluster_map, precomputed_strategies=strategies,
                                                                                                  precomputed_temperatures=temperatures,
                                                                                                  show_pids=False)

    print(keys)
    print("STRATEGY", strategy_proportions)
    print("CLUSTER", cluster_proportions)
    print("DECISION", decision_system_proportions)
