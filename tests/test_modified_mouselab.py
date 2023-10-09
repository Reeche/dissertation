import unittest
from pathlib import Path

import numpy as np
import yaml
from parameterized import parameterized

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.global_vars import structure
from mcl_toolbox.utils.learning_utils import Participant

"""
Tests modified (smaller) feature model
python3 -m unittest tests.test_modified_mouselab
"""

parameters = [
    # exp_name, pid, trial, feature_subset, expected_feature_array
    ["F1", 5, 2, "depth", np.asarray([[1, 0, 0, 1], [1, 1, 0, 1], [0, 2, 2, 1]])],
    [
        "F1",
        20,
        5,
        "depth",
        np.asarray(
            [
                [3, 0, 0, 1],
                [2, 1, 0, 1],
                [3, 2, 0, 1],
                [1, 3, 0, 1],
                [1, 4, 0, 1],
                [2, 5, 0, 1],
                [3, 6, 0, 1],
                [3, 7, 0, 1],
                [3, 8, 0, 1],
                [3, 9, 0, 1],
                [2, 10, 0, 1],
                [1, 11, 0, 1],
                [0, 12, 36, 1],
            ]
        ),
    ],
    ["F1", 151, 20, "depth", np.asarray([0, 0, 0, 1])],
]


def load_env_state_for_participant(exp_name, pid, trial):
    participant = Participant(
        exp_name, pid, excluded_trials=None, get_strategies=False, get_weights=False
    )
    pipeline = structure.exp_pipelines[exp_name]
    env = GenericMouselabEnv(
        len(participant.envs), pipeline=pipeline, ground_truth=participant.envs
    )

    # make sure actions in num trials match up
    assert len(participant.all_trials_data["actions"]) == env.num_trials

    for trial_idx in range(trial):
        env.get_next_trial()
        actions = participant.all_trials_data["actions"][trial_idx]

    return env, actions


def get_features(feature_subset):
    inputs_path = Path(__file__).parent.absolute().joinpath("inputs")
    yaml_input = inputs_path.joinpath(f"{feature_subset}.yaml")
    with open(yaml_input, "r") as stream:
        features = yaml.safe_load(stream)["features"]
    return features


class TestModifiedMouselab(unittest.TestCase):
    @parameterized.expand(parameters)
    def test_unnormalized_feature_values(
        self, exp_name, pid, trial, feature_subset, expected_feature_array
    ):
        features = get_features(feature_subset)
        env, actions = load_env_state_for_participant(exp_name, pid, trial)

        resulting_features = np.zeros((len(actions), len(features)))
        for action_idx, action in enumerate(actions):
            computed_features = env.present_trial.node_map[
                action
            ].compute_termination_feature_values(features, adaptive_satisficing={})
            resulting_features[action_idx, :] = computed_features
            _, _, done, _ = env.step(action)

        self.assertTrue(np.all(expected_feature_array == resulting_features))
