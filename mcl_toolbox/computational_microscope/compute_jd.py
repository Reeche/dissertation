import sys

import numpy as np

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import (create_dir, get_modified_weights,
                                              get_normalized_features,
                                              pickle_load, pickle_save)
from mcl_toolbox.utils.sequence_utils import compute_log_likelihood


def get_strategy_weights(weights_path, num_features, num_strategies):
    strategy_weights = np.zeros((num_strategies, num_features))
    for i in range(num_strategies):
        strategy_weights[i] = pickle_load(f"{weights_path}/{i+1}_20000.pkl")
    return strategy_weights


def compute_llk(f_s, s_s, f_clicks, trial_sequence, features, W, normalized_features):
    clicks_llk = []
    for clicks, trial in zip(f_clicks, trial_sequence):
        f_lik = compute_log_likelihood(
            trial, clicks, features, W[f_s - 1], normalized_features=normalized_features
        )
        s_lik = compute_log_likelihood(
            trial, clicks, features, W[s_s - 1], normalized_features=normalized_features
        )
        clicks_llk.append([f_lik, s_lik])
    return clicks_llk


def compute_jeffrey_kld(clicks_llk):
    clicks_llk = np.array(clicks_llk)
    return np.mean(clicks_llk[:, 0] - clicks_llk[:, 1])


def get_jeffrey_divergence(
    f_s,
    s_s,
    f_clicks,
    s_clicks,
    trial_sequence,
    features,
    W,
    normalized_features,
    num_simulations=1000,
):
    f_clicks_llk = compute_llk(
        f_s, s_s, f_clicks, trial_sequence, features, W, normalized_features
    )
    s_clicks_llk = compute_llk(
        s_s, f_s, s_clicks, trial_sequence, features, W, normalized_features
    )
    return compute_jeffrey_kld(f_clicks_llk) + compute_jeffrey_kld(s_clicks_llk)


if __name__ == "__main__":
    f_s = int(sys.argv[1]) + 1
    exp_num = sys.argv[2]
    s_s = int(sys.argv[3]) + 1
    num_simulations = int(sys.argv[4])

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

    C1, E = pickle_load(
        f"results/{exp_num}/strategy_clicks/{f_s}_{num_simulations}.pkl"
    )
    C2, E = pickle_load(
        f"results/{exp_num}/strategy_clicks/{s_s}_{num_simulations}.pkl"
    )

    num_simulations = len(E)
    env = GenericMouselabEnv(num_simulations, pipeline, ground_truth=E)
    trial_sequence = env.trial_sequence.trial_sequence
    jd = get_jeffrey_divergence(
        f_s,
        s_s,
        C1,
        C2,
        trial_sequence,
        features,
        W,
        normalized_features,
        num_simulations=num_simulations,
    )
    print(jd)
    create_dir(f"results/JD/{exp_num}")
    pickle_save(jd, f"results/JD/{exp_num}/{f_s}_{s_s}_{num_simulations}.pkl")
    pickle_save(jd, f"results/JD/{exp_num}/{s_s}_{f_s}_{num_simulations}.pkl")
