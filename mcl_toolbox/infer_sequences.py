import sys
import os
import random
from pathlib import Path
from mcl_toolbox.computational_microscope.computational_microscope import (
    ComputationalMicroscope,
)
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.global_vars import structure, strategies, features
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

"""
Run this file to infer the averaged sequences of the participants. 
Format: python3 infer_sequences.py <exp name> <num_trials> <block>
Example: python3 mcl_toolbox/infer_sequences.py T1.1 35 training
"""


def infer_experiment_sequences(
    exp_num="F1", num_trials=35, block="training", pids=None, max_evals=50, **kwargs
):
    """
    Infer the averaged sequences of the participants in an experiment.
    :param exp_num: experiment name, e.g. F1
    :param block: block, e.g. "training" or "test"
    :param pids: list of participants to use, otherwise None. could be useful to exclude paticipants in dataframe.
    :param max_evals: max optimization evals for fmin
    :return: strategy and temperature dicts with a key for every pid in the experiment. The strategies are a list for each trial, the temperature is temperature over all trials.
    Saves data either to results/inferred_strategies/<exp_num> or results/inferred_strategies/<exp_num_block>
    """
    # Initializations

    # 79 strategies out of 89
    strategy_space = strategies.strategy_space
    # no habitual features because each trial is considered individually
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    # For the new experiment that are not either v1.0, c1.1, c2.1_dec, F1 or IRL1
    if exp_num not in ["v1.0", "c1.1", "c2.1", "c2.1_dec", "F1", "IRL1"]:
        reward_dist = "categorical"
        reward_structure = exp_num
        reward_distributions = learning_utils.construct_reward_function(
            structure.reward_levels[reward_structure], reward_dist
        )
        repeated_pipeline = learning_utils.construct_repeated_pipeline(
            structure.branchings[exp_num], reward_distributions, num_trials
        )
        exp_pipelines = {exp_num: repeated_pipeline}
    else:
        # list of all experiments, e.g. v1.0, T1.1 only has the transfer after training (20 trials)
        exp_pipelines = structure.exp_pipelines
        if exp_num not in structure.exp_reward_structures:
            raise (ValueError, "Reward structure not found.")
        reward_structure = structure.exp_reward_structures[exp_num]

    if exp_num not in exp_pipelines:
        raise (ValueError, "Experiment pipeline not found.")
    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(num_trials+1)]

    # normalized_features = learning_utils.get_normalized_features(exp_num)
    normalized_features = learning_utils.get_normalized_features("v1.0")
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(
        pipeline,
        strategy_space,
        W,
        microscope_features,
        normalized_features=normalized_features,
    )

    # # TODO info on c2.1_dec should probably be added in global_vars, I also had a script I used to IRL
    # if exp_num == "c2.1_dec":
    #     # exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
    #     exp = Experiment("c2.1", cm=cm, pids=pids, block=block, data_path="../results/cm")
    # else:
    exp = Experiment(exp_num, cm=cm, pids=pids, block=block, data_path="../results/cm")
    exp.infer_strategies(max_evals=max_evals, show_pids=True)

    # create save path
    parent_directory = Path(__file__).parents[1]
    save_path = os.path.join(
        parent_directory, f"results/cm/inferred_strategies/{exp_num}"
    )
    # save_path = f"../results/cm/inferred_strategies/{exp_num}"
    if block:
        save_path += f"_{block}"
    learning_utils.create_dir(save_path)
    # save strategies, and temperatures
    inferred_strategies = exp.participant_strategies
    inferred_temperatures = exp.participant_temperatures
    learning_utils.pickle_save(inferred_strategies, f"{save_path}/strategies.pkl")
    learning_utils.pickle_save(inferred_temperatures, f"{save_path}/temperatures.pkl")

    return inferred_strategies, inferred_temperatures


if __name__ == "__main__":
    # random.seed(123)
    # exp_name = sys.argv[1]  # e.g. c2.1_dec
    # block = None
    # number_of_trials = int(sys.argv[2])
    # if len(sys.argv) > 2:
    #     block = sys.argv[3]

    exp_name = "strategy_discovery"
    number_of_trials = 120
    block = "training"

    # exp_name = "mf"
    # block = "training"

    # if exp_name == "mf":
    #     number_of_trials = 30
    # else:
    #     number_of_trials = 15

    infer_experiment_sequences(
        exp_name, number_of_trials, block, max_evals=100
    )  # max_evals have to be at least 2 for testing
