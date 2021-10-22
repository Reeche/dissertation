import sys

from mcl_toolbox.computational_microscope.computational_microscope import \
    ComputationalMicroscope
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import (create_dir, get_modified_weights,
                                              get_normalized_features,
                                              pickle_load, pickle_save)

if __name__ == "__main__":
    pid = int(sys.argv[1])
    mod_exp_num = sys.argv[2]
    block = None  # Change this if you have a particular block that you want to infer sequences for
    pids = [pid]

    strategy_space = pickle_load("data/strategy_space.pkl")
    features = pickle_load("data/microscope_features.pkl")
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    num_features = len(features)
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = {
        "v1.0": "high_increasing",
        "F1": "high_increasing",
        "c1.1_old": "low_constant",
        "T1.1": "large_increasing",
    }

    if mod_exp_num not in exp_pipelines:
        if mod_exp_num not in ["c2.1_inc", "c2.1_dec"]:
            exp_num = "v1.0"
            reward_structure = "high_increasing"

    additional_constraints = {}
    if exp_num not in ["c2.1_inc", "c2.1_dec"]:
        pipeline = [exp_pipelines[exp_num][0]] * 100
        reward_structure = exp_reward_structures[exp_num]
        if len(sys.argv) > 3:
            if exp_num == "F1":
                additional_constraints["condition"] = int(sys.argv[3])
            elif exp_num == "v1.0":
                additional_constraints["feedback"] = sys.argv[3]
    else:
        print(pids)
        if exp_num == "c2.1_inc":
            pipeline = [exp_pipelines["c2.1_inc"][0]] * 100
            reward_structure = "high_increasing"
            additional_constraints["variance"] = 2424
        else:
            pipeline = [exp_pipelines["c2.1_dec"][0]] * 100
            reward_structure = "high_decreasing"
            additional_constraints["variance"] = 2442
        mod_exp_num = "c2.1"

    normalized_features = get_normalized_features(reward_structure)
    W = get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(
        pipeline, strategy_space, W, features, normalized_features=normalized_features
    )
    v1 = Experiment(
        mod_exp_num, cm=cm, pids=pids, block=block, **additional_constraints
    )
    v1.infer_strategies(max_evals=100)
    create_dir(f"results/final_strategy_inferences/{exp_num}_{block}")
    strategies = pickle_save(
        v1.participant_strategies[pid],
        f"results/final_strategy_inferences/{exp_num}_{block}/{pid}_strategies.pkl",
    )
    temperatures = pickle_save(
        v1.participant_temperatures[pid],
        f"results/final_strategy_inferences/{exp_num}_{block}/{pid}_temperatures.pkl",
    )
    print(v1.participant_strategies)
