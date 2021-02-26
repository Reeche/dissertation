import sys, os
from pathlib import Path

from mcl_toolbox.global_vars import structure, strategies, features
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.utils.experiment_utils import Experiment

"""
Run this file to infer the averaged sequences of the participants. 
Format: python3 infer_sequences.py <exp name> <block>
Example: python3 infer_sequences.py T1.1 training
"""

def infer_experiment_sequences(exp_num = "F1", block = "training", pids = None, max_evals = 2, **kwargs):
    '''
    Infer the averaged sequences of the participants in an experiment.
    :param exp_num: experiment name, e.g. F1
    :param block: block, e.g. "training" or "test"
    :param pids: list of participants to use, otherwise None. could be useful to exclude paticipants in dataframe.
    :param max_evals: max optimization evals for fmin
    :return: strategy and temperature dicts with a key for every pid in the experiment. The strategies are a list for each trial, the temperature is temperature over all trials.
    Saves data either to results/inferred_strategies/<exp_num> or results/inferred_strategies/<exp_num_block>
    '''
    # Initializations

    # 79 strategies out of 89
    strategy_space = strategies.strategy_space
    # no habitual features because each trial is considered individually
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
    pipeline = [pipeline[0] for _ in range(100)]  # todo: why range 100?

    normalized_features = learning_utils.get_normalized_features(reward_structure)  # tuple of 2
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, microscope_features, normalized_features=normalized_features)

    #TODO info on c2.1_dec should probably be added in global_vars, I also had a script I used to IRL
    if exp_num == "c2.1_dec":
        # exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    exp.infer_strategies(max_evals=max_evals, show_pids=True)

    #create save path
    parent_directory = Path(__file__).parents[1]
    save_path = os.path.join(parent_directory, f"results/inferred_strategies/{exp_num}")
    if block:
        save_path += f"_{block}"
    learning_utils.create_dir(save_path)

    #save strategies, and temperatures
    inferred_strategies = exp.participant_strategies
    inferred_temperatures = exp.participant_temperatures
    learning_utils.pickle_save(inferred_strategies, f"{save_path}/strategies.pkl")
    learning_utils.pickle_save(inferred_temperatures, f"{save_path}/temperatures.pkl")

    return inferred_strategies, inferred_temperatures


if __name__ == "__main__":
    exp_name = sys.argv[1]  # increasing_variance, decreasing_variance
    block = None
    if len(sys.argv) > 2:
        block = sys.argv[2]

    infer_experiment_sequences(exp_name, block=block)