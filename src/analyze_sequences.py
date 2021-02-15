import sys
import numpy as np
from utils import learning_utils, distributions
sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
# from src.utils.learning_utils import pickle_load, get_normalized_features,\
#                             get_modified_weights
from computational_microscope.computational_microscope import ComputationalMicroscope
from utils.experiment_utils import Experiment

"""
Run this file to analyse the inferred sequences of the participants. 
Format: python3 analyze_sequences.py <reward_structure> <block> <pid>
Example: python3 analyze_sequences.py increasing_variance training none
"""

if __name__ == "__main__":
    # reward_structure = sys.argv[1]
    # block = None
    # if len(sys.argv) > 2:
    #     block = sys.argv[2]
    reward_structure = "increasing_variance"
    block ="training"

    # Initializations
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

    exp_reward_structures = {'increasing_variance': 'high_increasing',
                             'constant_variance': 'low_constant',
                             'decreasing_variance': 'high_decreasing',
                             'transfer_task': 'large_increasing'}

    reward_exps = {"increasing_variance": "v1.0",
                   "decreasing_variance": "c2.1_dec",
                   "constant_variance": "c1.1",
                   "transfer_task": "T1.1"}

    exp_num = reward_exps[reward_structure]
    if exp_num not in exp_pipelines:
        raise (ValueError, "Reward structure not found.")

    pipeline = exp_pipelines[exp_num]
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = learning_utils.get_normalized_features(exp_reward_structures[reward_structure])
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)

    pids = None
    if exp_num == "c2.1_dec":
        # exp = Experiment("c2.1", cm=cm, pids=pids, block = block, variance = 2442)
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    dir_path = f"../results/inferred_strategies/{reward_structure}"
    if block:
        dir_path += f"_{block}"

    try:
        strategies = learning_utils.pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = learning_utils.pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print("Exception", e)
        # exit()

    if exp_num == "c2.1_dec":
        save_path = f"../results/c2.1"
        if block:
            save_path += f"_{block}"
    else:
        save_path = f"../results/{exp_num}"
        if block:
            save_path += f"_{block}"
    learning_utils.create_dir(save_path)
    #print("DS", np.sum(DS_proportions, axis = 1))
    exp.summarize(features, normalized_features, strategy_weights,
                  decision_systems, W_DS, DS_proportions, strategy_scores,
                  cluster_scores, cluster_map, precomputed_strategies=strategies,
                  precomputed_temperatures=temperatures,
                  show_pids=False)