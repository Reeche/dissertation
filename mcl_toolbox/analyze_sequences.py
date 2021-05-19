import sys
import random
import os
from pathlib import Path
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

from mcl_toolbox.utils.learning_utils import pickle_load, get_normalized_features, get_modified_weights, create_dir
from mcl_toolbox.global_vars import structure, strategies, features
from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.utils.experiment_utils import Experiment

"""
Run this file to analyse the inferred sequences of the participants. 
Format: python3 analyze_sequences.py <reward_structure> <num_trials> <block> <create_plot>
Example: python3 analyze_sequences.py c2.1_dec 35 training True

Please remember to set a seed
"""


def analyse_sequences(exp_num="v1.0", num_trials=35, block="training", pids=None,
                      create_plot=True, number_of_top_worst_strategies=5, **kwargs):
    """
    Creates plots. For details of the args, check out global_vars.py
    Args:
        exp_num: experiment number, str e.g. "v1.0"
        num_trials: int, e.g. 35
        block: training or test, str
        pids: participant id
        create_plot: boolean, do you want to create plots
        number_of_top_worst_strategies: for creating aggregated adaptive and maladaptive plots. How many good/bad ones do you want to aggregate?
        **kwargs:

    Returns:

    """
    # Initializations
    decision_systems = pickle_load("../data/decision_systems.pkl")
    DS_proportions = pickle_load("../data/strategy_decision_proportions.pkl")
    W_DS = pickle_load("../data/strategy_decision_weights.pkl")
    cluster_map = pickle_load("../data/kl_cluster_map.pkl")
    strategy_scores = pickle_load(
        "../data/strategy_scores.pkl")  # todo: update strategy scores to contain all environments, currently only increasing variance
    cluster_scores = pickle_load("../data/cluster_scores.pkl")

    strategy_space = strategies.strategy_space
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    # For the new experiment that are not either v1.0, c1.1, c2.1_dec, F1 or IRL1
    if exp_num not in ["v1.0", "c1.1", "c2.1_dec"]:
        reward_dist = "categorical"
        reward_structure = exp_num
        reward_distributions = learning_utils.construct_reward_function(structure.reward_levels[reward_structure],
                                                                        reward_dist)
        repeated_pipeline = learning_utils.construct_repeated_pipeline(structure.branchings[exp_num],
                                                                       reward_distributions,
                                                                       num_trials)
        exp_pipelines = {exp_num: repeated_pipeline}
    else:
        exp_pipelines = structure.exp_pipelines
        if exp_num not in structure.exp_reward_structures:
            raise (ValueError, "Reward structure not found.")
        reward_structure = structure.exp_reward_structures[exp_num]

    if exp_num not in exp_pipelines:
        raise (ValueError, "Experiment pipeline not found.")
    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = get_normalized_features(reward_structure)
    W = get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, microscope_features,
                                 normalized_features=normalized_features)
    pids = None
    if exp_num == "c2.1_dec":
        # exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    parent_directory = Path(__file__).parents[1]
    dir_path = os.path.join(parent_directory, f"results/cm/inferred_strategies/{exp_num}")

    # dir_path = f"../results/cm/inferred_strategies/{exp_num}"

    if block:
        dir_path += f"_{block}"

    try:
        strategies_ = pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print("Exception", e)
        # exit()

    if exp_num == "c2.1_dec":
        save_path = f"../results/cm/plots/c2.1"
        if block:
            save_path += f"_{block}"
    else:
        save_path = f"../results/cm/plots/{exp_num}"
        if block:
            save_path += f"_{block}"
    create_dir(save_path)

    if create_plot:
        exp.summarize(
            features, normalized_features, strategy_weights,
            decision_systems, W_DS, DS_proportions, strategy_scores,
            cluster_scores, cluster_map,
            number_of_top_worst_strategies=number_of_top_worst_strategies,
            create_plot=create_plot,
            precomputed_strategies=strategies_,
            precomputed_temperatures=temperatures,
            show_pids=False)
    else:
        strategy_proportions, strategy_proportions_trialwise, cluster_proportions, cluster_proportions_trialwise, \
        decision_system_proportions, mean_dsw, adaptive_strategies_proportion, maladaptive_strategies_proportion, number_of_clicks, adaptive_participants, \
        maladaptive_participants, other_participants, improved_participants = exp.summarize(
            features, normalized_features, strategy_weights,
            decision_systems, W_DS, DS_proportions, strategy_scores,
            cluster_scores, cluster_map,
            number_of_top_worst_strategies=number_of_top_worst_strategies,
            create_plot=create_plot,
            precomputed_strategies=strategies_,
            precomputed_temperatures=temperatures,
            show_pids=False)
        return strategy_proportions, strategy_proportions_trialwise, cluster_proportions, cluster_proportions_trialwise, \
               decision_system_proportions, mean_dsw, adaptive_strategies_proportion, maladaptive_strategies_proportion, \
               number_of_clicks, adaptive_participants, maladaptive_participants, other_participants, improved_participants


if __name__ == "__main__":
    # random.seed(123)
    # exp_name = sys.argv[1]  # e.g. c2.1_dec
    # block = None
    # if len(sys.argv) > 2:
    #     number_of_trials = int(sys.argv[2])
    #     block = sys.argv[3]

    exp_name = "c1.1"
    block = "training"
    number_of_trials = 35
    create_plot = True

    # create the plots
    analyse_sequences(exp_name, number_of_trials, block, create_plot, number_of_top_worst_strategies=5)
