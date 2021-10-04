from pathlib import Path

import numpy as np
from costometer.utils import create_mcrl_reward_distribution
from mouselab.envs.registry import registry

from mcl_toolbox.env.modified_mouselab import TrialSequence
from mcl_toolbox.global_vars import strategies
from mcl_toolbox.utils.learning_utils import construct_repeated_pipeline, pickle_save
from mcl_toolbox.utils.planning_strategies import strategy_dict
from mcl_toolbox.utils.sequence_utils import compute_trial_features


def contruct_pipeline(branching, reward_function, num_trials=30):
    return [(branching, reward_function)] * num_trials


def generate_data(strategy_num, pipeline, num_trials=30):
    env = TrialSequence(len(pipeline), pipeline)
    ground_truth = env.ground_truth
    simulated_actions = []
    for trial_num in range(num_trials):
        trial = env.trial_sequence[trial_num]
        actions = strategy_dict[strategy_num](trial)
        simulated_actions.append(actions)
    return ground_truth, simulated_actions


def normalize(pipeline, features_list):
    simulated_features = []
    for strategy_num in strategies.strategy_space:
        ground_truth, simulated_actions = generate_data(
            strategy_num, pipeline, len(pipeline)
        )
        for trial_num, _ in enumerate(pipeline):
            trial_actions = simulated_actions[trial_num]
            trial_features = compute_trial_features(
                pipeline, ground_truth[trial_num], trial_actions, features_list, False
            )
            simulated_features += trial_features.tolist()
    simulated_features = np.array(simulated_features)
    simulated_features_shape = simulated_features.shape
    simulated_features = simulated_features.reshape(-1, simulated_features_shape[-1])
    max_feature_values = np.max(simulated_features, axis=0)
    min_feature_values = np.min(simulated_features, axis=0)
    max_feature_values = {f: max_feature_values[i] for i, f in enumerate(features_list)}
    min_feature_values = {f: min_feature_values[i] for i, f in enumerate(features_list)}
    return max_feature_values, min_feature_values


def get_new_feature_normalization(
    features_list, exp_setting="high_increasing", num_trials=30, num_simulations=100
):
    branching = registry(exp_setting).branching
    reward_distributions = create_mcrl_reward_distribution(exp_setting)
    pipeline = construct_repeated_pipeline(branching, reward_distributions, num_trials)
    max_fv, min_fv = normalize(pipeline, features_list)
    return max_fv, min_fv


def save_normalized_values(max_fv, min_fv, exp_name):
    save_path = Path(__file__).parents[1].joinpath(f"data/normalized_values/{exp_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    pickle_save(max_fv, str(save_path.joinpath("max.pkl")))
    pickle_save(min_fv, str(save_path.joinpath("min.pkl")))


if __name__ == "__main__":
    features_list = ["depth", "num_clicks", "return_if_terminating", "constant"]
    exp_name = "DepthSubset"

    max_fv, min_fv = get_new_feature_normalization(
        features_list, exp_setting="high_increasing", num_trials=30, num_simulations=100
    )
    save_normalized_values(max_fv, min_fv, exp_name)
