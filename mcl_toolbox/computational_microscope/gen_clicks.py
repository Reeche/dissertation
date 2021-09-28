import sys

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import (create_dir, get_modified_weights,
                                              get_normalized_features,
                                              pickle_load, pickle_save)
from mcl_toolbox.utils.sequence_utils import get_clicks


def gen_envs(pipeline):
    num_simulations = len(pipeline)
    env = GenericMouselabEnv(num_simulations, pipeline)
    E = env.ground_truth
    return E


def GCE(E, pipeline, features, w, normalized_features):
    envs = E
    num_simulations = len(pipeline)
    env = GenericMouselabEnv(num_simulations, pipeline, ground_truth=envs)
    trial_sequence = env.trial_sequence.trial_sequence
    C = []
    for trial in trial_sequence:
        C.append(get_clicks(trial, features, w, normalized_features))
        trial.reset_observations()
    return C


if __name__ == "__main__":
    strategy_num = int(sys.argv[1])
    exp_num = sys.argv[2]
    num_simulations = int(sys.argv[3])
    num_strategies = 89
    strategy_space = list(range(1, num_strategies + 1))
    features = pickle_load("data/microscope_features.pkl")
    num_features = len(features)

    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = {
        "v1.0": "high_increasing",
        "F1": "high_increasing",
        "c1.1_old": "low_constant",
        "T1.1": "large_increasing",
    }

    # Defaults for 312 increasing variance task
    reward_structure = "high_increasing"
    pipeline = [exp_pipelines["v1.0"][0]] * num_simulations

    if exp_num in exp_reward_structures:
        reward_structure = exp_reward_structures[exp_num]
        pipeline = [exp_pipelines[exp_num][0]] * num_simulations
    else:
        if exp_num == "c2.1_inc":
            reward_structue = "high_increasing"
            pipeline = [exp_pipelines["c2.1_inc"][0]] * num_simulations
        else:
            reward_structure = "high_decreasing"
            pipeline = [exp_pipelines["c2.1_dec"][0]] * num_simulations

    normalized_features = get_normalized_features(reward_structure)
    strategy_weights = pickle_load("data/microscope_weights.pkl")
    W = get_modified_weights(strategy_space, strategy_weights)
    w = W[strategy_num]

    env_dir = f"results/cluster_envs/{exp_num}"
    E = pickle_load(f"{env_dir}/random_envs_{num_simulations}.pkl")
    C = GCE(E, pipeline, features, w, normalized_features)

    dir_path = f"results/{exp_num}/strategy_clicks"
    create_dir(dir_path)
    pickle_save((C, E), f"{dir_path}/{strategy_num+1}_{num_simulations}.pkl")