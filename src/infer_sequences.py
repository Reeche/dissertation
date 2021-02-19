import sys
from utils import learning_utils, distributions
sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
from computational_microscope.computational_microscope import ComputationalMicroscope
from utils.experiment_utils import Experiment

"""
Run this file to infer the averaged sequences of the participants. 
Format: python3 infer_sequences.py <reward_structure> <block> <pid>
Example: python3 infer_sequences.py increasing_variance training none
"""


if __name__ == "__main__":
    reward_structure = sys.argv[1]  # increasing_variance, decreasing_variance
    block = None
    if len(sys.argv) > 2:
        block = sys.argv[2]
    # reward_structure = "increasing_variance"
    # block = "training"


    # Initializations
    strategy_space = learning_utils.pickle_load("data/strategy_space.pkl") #79 strategies out of 89
    features = learning_utils.pickle_load("data/microscope_features.pkl") #no habitual features because each trial is considered individually
    strategy_weights = learning_utils.pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")  # list of all experiments, e.g. v1.0, T1.1 only has the transfer after training (20 trials)
    exp_reward_structures = {'increasing_variance': 'high_increasing',
                             'constant_variance': 'low_constant',
                             'decreasing_variance': 'high_decreasing',
                             'transfer_task': 'large_increasing'}

    reward_exps = {"increasing_variance": "v1.0",
                   "decreasing_variance": "c2.1_dec",
                   "constant_variance": "c1.1",
                   "transfer_task": "T1.1"}

    exp_num = reward_exps[reward_structure]  # select experiment number, e.g. v1.0 given entered selection
    if exp_num not in exp_pipelines:
        raise (ValueError, "Reward structure not found.")

    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(100)]  # todo: why range 100?

    normalized_features = learning_utils.get_normalized_features(exp_reward_structures[reward_structure])  # tuple of 2
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)
    pids = None
    if exp_num == "c2.1_dec":
        #exp = Experiment("c2.1", cm=cm, pids=pids, block=block, variance=2442)
        exp = Experiment("c2.1", cm=cm, pids=pids, block=block)
    else:
        exp = Experiment(exp_num, cm=cm, pids=pids, block=block)
    exp.infer_strategies(max_evals=2, show_pids=True)

    save_path = f"../results/inferred_strategies/{reward_structure}"
    if block:
        save_path += f"_{block}"
    learning_utils.create_dir(save_path)
    strategies = exp.participant_strategies
    temperatures = exp.participant_temperatures
    learning_utils.pickle_save(strategies, f"{save_path}/strategies.pkl")
    learning_utils.pickle_save(temperatures, f"{save_path}/temperatures.pkl")
