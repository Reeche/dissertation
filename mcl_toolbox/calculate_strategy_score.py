import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.planning_strategies import strategy_dict
from mcl_toolbox.global_vars import structure
from mcl_toolbox.utils import learning_utils, distributions

# For backward compatibility with pickled objects
sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions


def click_sequence_cost(click_sequence):
    """Returns the total cost of a sequence of clicks."""

    def cost(click):
        if click in [0]:
            return 0
        elif click in [1, 5, 9]:
            return 1
        elif click in [2, 6, 10]:
            return 3
        elif click in [3, 4, 7, 8, 11, 12]:
            return 30
        else:
            raise ValueError(f"Unexpected click value: {click}")

    return sum(cost(c) for c in click_sequence)


def load_pipeline(exp_name, reward_level, num_trials):
    """Loads or constructs a pipeline based on the experiment type."""
    if exp_name in ["v1.0", "c1.1", "c2.1_dec"]:
        return learning_utils.pickle_load("data/exp_pipelines.pkl")

    reward_dist = "categorical"
    reward_distributions = learning_utils.construct_reward_function(
        structure.reward_levels[reward_level], reward_dist
    )
    repeated_pipeline = learning_utils.construct_repeated_pipeline(
        structure.branchings[exp_name], reward_distributions, num_trials
    )
    return {exp_name: repeated_pipeline}


def simulate_strategy(strategy_id, pipeline, exp_name, num_simulations, click_cost):
    """Runs the strategy simulation for a given strategy ID."""
    scores = []
    clicks_counts = []

    for _ in range(num_simulations):
        env = GenericMouselabEnv(num_trials=1, pipeline=pipeline)
        click_sequence = strategy_dict[strategy_id](env.present_trial)
        reward = env.present_trial.node_map[0].calculate_max_expected_return()
        cost = click_sequence_cost(click_sequence)
        score = reward - cost

        scores.append(score)
        clicks_counts.append(len(click_sequence))

    return np.mean(scores), np.mean(clicks_counts)


def simulate_all_strategies(exp_name, num_simulations, click_cost, reward_level, num_trials=35):
    """Simulates all strategies and collects average scores and click counts."""
    exp_pipelines = load_pipeline(exp_name, reward_level, num_trials)
    pipeline = exp_pipelines[exp_name]

    strategy_scores = {}
    click_counts = {}

    for strategy_id in range(1, 90):  # strategy_dict keys are 1-indexed
        print(f"Simulating strategy {strategy_id}...")
        avg_score, avg_clicks = simulate_strategy(
            strategy_id, pipeline, exp_name, num_simulations, click_cost
        )
        print(f"Score: {avg_score:.4f}, Clicks: {avg_clicks:.2f}")
        strategy_scores[strategy_id - 1] = avg_score
        click_counts[strategy_id - 1] = avg_clicks

    return strategy_scores, click_counts


def save_results(exp_name, strategy_scores, click_counts):
    """Saves strategy scores and click counts to disk."""
    result_dir = Path(f"../results/cm/strategy_scores/strategy_discovery")
    learning_utils.create_dir(result_dir)

    score_path = result_dir / f"{exp_name}_clickcost_strategy_scores.pkl"
    click_path = result_dir / f"{exp_name}_clickcost_numberclicks.pkl"

    learning_utils.pickle_save(dict(sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)), score_path)
    learning_utils.pickle_save(click_counts, click_path)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 calculate_strategy_score.py <exp_name> <num_simulations> <click_cost> <reward_level>")
        sys.exit(1)

    exp_name = sys.argv[1]
    num_simulations = int(sys.argv[2])
    click_cost = int(sys.argv[3])
    reward_level = sys.argv[4]

    strategy_scores, click_counts = simulate_all_strategies(
        exp_name, num_simulations, click_cost, reward_level
    )

    save_results(exp_name, strategy_scores, click_counts)
