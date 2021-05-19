import sys
import os
from pathlib import Path

from mcl_toolbox.global_vars import structure, strategies, features
from mcl_toolbox.utils import learning_utils

from mcl_toolbox.computational_microscope.computational_microscope import (
    ComputationalMicroscope,
)
from mcl_toolbox.utils.analysis_utils import get_data

"""
Run this file to infer the averaged(?) sequences of the participants.
Format: python3 infer_sequences.py <pid> <reward_structure> <block>
Example: python3 infer_participant_sequences.py 1 v1.0 training
"""


def modify_clicks(click_sequence):
    modified_clicks = []
    for clicks in click_sequence:
        modified_clicks.append([int(c) for c in clicks] + [0])
    return modified_clicks


def get_participant_data(exp_num, pid, block=None):
    data = get_data(exp_num)
    clicks_data = data["mouselab-mdp"]
    if block:
        clicks_data = clicks_data[
            (clicks_data.pid == pid) & (clicks_data.block == block)
        ]
    else:
        clicks_data = clicks_data[clicks_data.pid == pid]
    click_sequence = [q["click"]["state"]["target"] for q in clicks_data.queries]
    click_sequence = modify_clicks(click_sequence)
    if "stateRewards" in clicks_data.columns:
        envs = [[0] + sr[1:] for sr in clicks_data.stateRewards]
    elif "state_rewards" in clicks_data.columns:
        envs = [[0] + sr[1:] for sr in clicks_data.state_rewards]
    return click_sequence, envs


def infer_strategies(
    click_sequences, envs, pipeline, strategy_space, W, features, normalized_features
):
    cm = ComputationalMicroscope(
        pipeline, strategy_space, W, features, normalized_features
    )
    S, _, _, T = cm.infer_sequences(click_sequences, envs)
    return S, T


def infer_many_participant_sequences(
    exp_num_list=["F1"], pid_list=[0, 3, 5, 13], block="training"
):
    """
    Infers each participant individually, for a list of experiments
    :param exp_num_list: list of experiment names
    :param pid_list: list of pids as integers
    :param block: block name, None if we want to look at all
    :return: saves inferred strategy and temperature files for each participant in
    results/inferred_participant_sequences/{exp_num}
    """
    for exp in exp_num_list:
        for pid in pid_list:
            infer_participant_sequences(pid, exp, block)


def infer_participant_sequences(pid, exp_num, block=None):
    strategy_space = strategies.strategy_space
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    exp_pipelines = structure.exp_pipelines

    if exp_num not in structure.exp_reward_structures:
        raise (ValueError, "Reward structure not found.")
    reward_structure = structure.exp_reward_structures[exp_num]

    if exp_num not in exp_pipelines:
        raise (ValueError, "Experiment pipeline not found.")
    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = learning_utils.get_normalized_features(reward_structure)
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)

    # Get clicks and envs of a particular participant
    clicks, envs = get_participant_data(exp_num, pid, block=block)
    # TODO these analysis files are switched around,
    # the one uses ComputationalMicroscope.infer_strategies, whereas infer_sequences.py uses ComputationalMicroscope.infer_participant_sequences
    # I see why they might've been named this as infer_sequences works for a whole experiment, whereas this one is participant by participant
    inferred_strategies, inferred_temperatures = infer_strategies(
        clicks,
        envs,
        pipeline,
        strategy_space,
        W,
        microscope_features,
        normalized_features,
    )

    parent_directory = Path(__file__).parents[1]
    path = os.path.join(
        parent_directory, f"results/inferred_participant_sequences/{exp_num}"
    )
    learning_utils.create_dir(path)
    if not block:
        learning_utils.pickle_save(inferred_strategies, f"{path}/{pid}_strategies.pkl")
        learning_utils.pickle_save(
            inferred_temperatures, f"{path}/{pid}_temperature.pkl"
        )
    else:
        learning_utils.pickle_save(
            inferred_strategies, f"{path}/{pid}_{block}_strategies.pkl"
        )
        learning_utils.pickle_save(
            inferred_temperatures, f"{path}/{pid}_{block}_temperature.pkl"
        )


if __name__ == "__main__":
    pid = int(sys.argv[1])
    exp_num = sys.argv[2]
    block = None
    if len(sys.argv) > 3:
        block = sys.argv[3]
    infer_participant_sequences(pid, exp_num, block)
