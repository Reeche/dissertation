import os
import sys
from pathlib import Path

from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.global_vars import features, strategies, structure
from mcl_toolbox.utils import learning_utils
from utils.analysis_utils import get_data


def modify_clicks(click_sequence):
    """Convert click targets to int and append a 0 to each."""
    return [[int(c) for c in clicks] + [0] for clicks in click_sequence]


def get_participant_clicks_and_envs(exp_name, pid, block=None):
    """Loads and prepares participant click sequences and reward environments."""
    data = get_data(exp_name)
    clicks_data = data["mouselab-mdp"]

    if block:
        clicks_data = clicks_data[(clicks_data.pid == pid) & (clicks_data.block == block)]
    else:
        clicks_data = clicks_data[clicks_data.pid == pid]

    click_sequences = [q["click"]["state"]["target"] for q in clicks_data.queries]
    click_sequences = modify_clicks(click_sequences)

    if "stateRewards" in clicks_data.columns:
        envs = [[0] + sr[1:] for sr in clicks_data.stateRewards]
    elif "state_rewards" in clicks_data.columns:
        envs = [[0] + sr[1:] for sr in clicks_data.state_rewards]
    else:
        raise ValueError("Missing reward column in dataset.")

    return click_sequences, envs


def get_pipeline(exp_name, num_trials):
    """Returns the environment pipeline for the given experiment."""
    if exp_name not in ["v1.0", "c1.1", "c2.1", "c2.1_dec", "F1", "IRL1"]:
        reward_distributions = learning_utils.construct_reward_function(
            structure.reward_levels[exp_name], "categorical"
        )
        repeated_pipeline = learning_utils.construct_repeated_pipeline(
            structure.branchings[exp_name], reward_distributions, num_trials
        )
        return [repeated_pipeline[0] for _ in range(num_trials + 1)]
    else:
        if exp_name not in structure.exp_pipelines:
            raise ValueError(f"No pipeline defined for experiment '{exp_name}'")
        return [structure.exp_pipelines[exp_name][0] for _ in range(num_trials + 1)]


def infer_strategies(click_sequences, envs, exp_name, pipeline, max_evals):
    """Infers strategies and temperatures using the Computational Microscope."""
    strategy_space = strategies.strategy_space
    strategy_weights = strategies.strategy_weights
    feature_set = features.microscope

    normalized_features = learning_utils.get_normalized_features(exp_name)
    weights = learning_utils.get_modified_weights(strategy_space, strategy_weights)

    cm = ComputationalMicroscope(
        pipeline,
        strategy_space,
        weights,
        feature_set,
        normalized_features=normalized_features
    )

    strategies_out, _, _, temperatures = cm.infer_sequences(
        click_sequences, envs, max_evals
    )
    return strategies_out, temperatures


def save_results(exp_name, pid, block, strategies, temperatures):
    """Saves inferred strategies and temperatures to disk."""
    parent_directory = Path(__file__).parents[1]
    output_dir = os.path.join(parent_directory, f"results/inferred_participant_sequences/{exp_name}")
    learning_utils.create_dir(output_dir)

    block_suffix = f"_{block}" if block else ""
    learning_utils.pickle_save(strategies, f"{output_dir}/{pid}{block_suffix}_strategies.pkl")
    learning_utils.pickle_save(temperatures, f"{output_dir}/{pid}{block_suffix}_temperature.pkl")


def infer_participant_sequences(pid, exp_name, num_trials, block=None, max_evals=100):
    """Main function for inferring participant strategies."""
    click_sequences, envs = get_participant_clicks_and_envs(exp_name, pid, block)
    pipeline = get_pipeline(exp_name, num_trials)

    strategies_out, temperatures = infer_strategies(
        click_sequences, envs, exp_name, pipeline, max_evals
    )

    save_results(exp_name, pid, block, strategies_out, temperatures)


def infer_multiple_participants(exp_list, pid_list, block="training", num_trials=120):
    """Batch process for multiple experiments and participants."""
    for exp_name in exp_list:
        for pid in pid_list:
            infer_participant_sequences(pid, exp_name, num_trials, block)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python infer_participant_sequences.py <pid> <exp_name> <num_trials> [<block>]")
        sys.exit(1)

    pid = int(sys.argv[1])
    exp_name = str(sys.argv[2])
    num_trials = int(sys.argv[3])
    block = sys.argv[4] if len(sys.argv) > 4 else None

    infer_participant_sequences(pid, exp_name, num_trials, block, max_evals=100)
