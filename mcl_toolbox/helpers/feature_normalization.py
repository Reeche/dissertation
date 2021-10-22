import numpy as np
from mcl_toolbox.utils.learning_utils import (
    create_dir,
    pickle_load,
    pickle_save,
    construct_repeated_pipeline,
)
from mcl_toolbox.env.modified_mouselab import (
    TrialSequence,
    reward_val,
    # normal_reward_val,
)
from mcl_toolbox.utils.planning_strategies import strategy_dict

strategy_space = pickle_load("../data/strategy_space.pkl")


def generate_data(strategy_num, pipeline, num_simulations=1000):
    env = TrialSequence(num_simulations, pipeline)
    ground_truth = env.ground_truth
    simulated_actions = []
    for sim_num in range(num_simulations):
        trial = env.trial_sequence[sim_num]
        actions = strategy_dict[strategy_num](trial)
        simulated_actions.append(actions)
    return ground_truth, simulated_actions


def compute_trial_features(pipeline, ground_truth, trial_actions, features_list):
    num_features = len(features_list)
    env = TrialSequence(num_trials=1, pipeline=pipeline, ground_truth=[ground_truth])
    trial = env.trial_sequence[0]
    num_actions = len(trial_actions)
    num_nodes = trial.num_nodes
    action_feature_values = np.zeros((num_actions, num_nodes, num_features))
    for i, action in enumerate(trial_actions):
        node_map = trial.node_map
        for node_num in range(num_nodes):
            node = trial.node_map[node_num]
            action_feature_values[i][
                node_num
            ] = node.compute_termination_feature_values(features_list)
        node_map[action].observe()
    return action_feature_values


def normalize(pipeline, features_list, num_simulations=1000):
    # num_strategies = len(strategy_space)
    simulated_features = []
    for strategy_num in strategy_space:
        ground_truth, simulated_actions = generate_data(
            strategy_num, pipeline, num_simulations
        )
        for sim_num in range(num_simulations):
            trial_actions = simulated_actions[sim_num]
            trial_features = compute_trial_features(
                pipeline, ground_truth[sim_num], trial_actions, features_list
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


if __name__ == "__main__":
    exp_num = "high_variance_low_cost"
    num_simulations = 1000
    features_list = pickle_load("../data/implemented_features.pkl")
    branching = [3, 1, 2]
    # for your case replace normal_reward_val with your own distribution or
    # just replace the pipeline variable with the pipeline you create for your new experiment
    pipeline = construct_repeated_pipeline(branching, reward_val, num_simulations)
    max_fv, min_fv = normalize(pipeline, features_list, num_simulations)
    # exp_branching = "_".join([str(b) for b in branching])
    dir_path = f"../data/normalized_values/{exp_num}"
    create_dir(dir_path)
    pickle_save(max_fv, dir_path + "/max.pkl")
    pickle_save(min_fv, dir_path + "/min.pkl")
    print(max_fv, min_fv)
