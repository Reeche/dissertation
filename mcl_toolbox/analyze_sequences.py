import random
import sys

from mcl_toolbox.global_vars import features, strategies, structure
from mcl_toolbox.utils import distributions, learning_utils

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

from mcl_toolbox.computational_microscope.computational_microscope import (
    ComputationalMicroscope,
)
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import create_dir, pickle_load

"""
Run this file to analyse the inferred sequences of the participants.
Format: python3 analyze_sequences.py <reward_structure> <block> <create_plot>
Example: python3 analyze_sequences.py c2.1_dec training True
"""


def analyse_sequences(
    exp_num="v1.0",
    block="training",
    pids=None,
    create_plot=False,
    number_of_top_worst_strategies=3,
    **kwargs,
):
    # Initializations
    decision_systems = learning_utils.pickle_load("data/decision_systems.pkl")
    DS_proportions = learning_utils.pickle_load(
        "data/strategy_decision_proportions.pkl"
    )
    W_DS = learning_utils.pickle_load("data/strategy_decision_weights.pkl")
    cluster_map = learning_utils.pickle_load("data/kl_cluster_map.pkl")
    strategy_scores = learning_utils.pickle_load(
        "data/strategy_scores.pkl"
    )  # todo: update strategy scores to contain all environments, currently only increasing variance
    cluster_scores = learning_utils.pickle_load("data/cluster_scores.pkl")

    strategy_space = strategies.strategy_space
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    # list of all experiments, e.g. v1.0, T1.1 only has the transfer after training (20 trials)
    exp_pipelines = structure.exp_pipelines
    if exp_num not in structure.exp_reward_structures:
        raise (ValueError, "Reward structure not found.")
    reward_structure = structure.exp_reward_structures[exp_num]

    if exp_num not in exp_pipelines:
        raise (ValueError, "Experiment pipeline not found.")
    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = learning_utils.get_normalized_features(
        reward_structure
    )  # tuple of 2
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(
        pipeline,
        strategy_space,
        W,
        microscope_features,
        normalized_features=normalized_features,
    )
    pids = None
    if exp_num == "c2.1_dec":
        # exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    dir_path = f"../../results/cm/inferred_strategies/{exp_num}"
    if block:
        dir_path += f"_{block}"

    try:
        strategies_ = pickle_load(f"{dir_path}/strategies.pkl")
        temperatures = pickle_load(f"{dir_path}/temperatures.pkl")
    except Exception as e:
        print("Exception", e)
        # exit()

    if exp_num == "c2.1_dec":
        save_path = "../results/cm/plots/c2.1"
        if block:
            save_path += f"_{block}"
    else:
        save_path = f"../results/cm/plots/{exp_num}"
        if block:
            save_path += f"_{block}"
    create_dir(save_path)

    if create_plot:
        exp.summarize(
            features,
            normalized_features,
            strategy_weights,
            decision_systems,
            W_DS,
            DS_proportions,
            strategy_scores,
            cluster_scores,
            cluster_map,
            number_of_top_worst_strategies=number_of_top_worst_strategies,
            create_plot=create_plot,
            precomputed_strategies=strategies_,
            precomputed_temperatures=temperatures,
            show_pids=False,
        )
    else:
        manual_strategy_list = []
        maladaptive_strategy_list = []

    exp.summarize(
        features,
        normalized_features,
        strategy_weights,
        decision_systems,
        W_DS,
        DS_proportions,
        strategy_scores,
        cluster_scores,
        cluster_map,
        manual_strategy_list,
        maladaptive_strategy_list,
        precomputed_strategies=strategies_,
        precomputed_temperatures=temperatures,
        show_pids=False,
    )


if __name__ == "__main__":
    random.seed(123)
    # exp_name = sys.argv[1]  # e.g. c2.1_dec
    # block = None
    # create_plot = True
    # if len(sys.argv) > 2:
    #     block = sys.argv[2]
    #     create_plot = sys.argv[3]

    exp_name = "c1.1"
    block = "training"
    create_plot = True

    # create the plots
    analyse_sequences(
        exp_name, block=block, create_plot=True, number_of_top_worst_strategies=4
    )
