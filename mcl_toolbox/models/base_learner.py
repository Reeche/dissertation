from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from mcl_toolbox.env.modified_mouselab import get_termination_mers
from mcl_toolbox.utils.learning_utils import get_normalized_feature_values


class Learner(ABC):
    """Base class of RL models implemented for the Mouselab-MDP paradigm."""

    def __init__(self, params, attributes):
        super().__init__()
        self.pr_weight = params["pr_weight"]
        if "delay_scale" in params:
            self.delay_scale = np.exp(params["delay_scale"])
        else:
            self.delay_scale = 0
        if "subjective_cost" in params:
            self.subjective_cost = params["subjective_cost"]
        else:
            self.subjective_cost = 0
        self.features = attributes["features"]
        self.num_features = len(self.features)
        self.normalized_features = attributes["normalized_features"]
        self.use_pseudo_rewards = attributes["use_pseudo_rewards"]
        self.is_null = attributes["is_null"]
        self.path_learn = False
        if "path_learn" in attributes:
            self.path_learn = attributes["path_learn"]
        self.previous_best_paths = []
        self.compute_likelihood = False

    @abstractmethod
    def simulate(self, env, compute_likelihood=False, participant=False):
        pass

    def get_term_reward(self, env):
        """Get the max expected return in the current state"""
        pres_node_map = env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    def get_term_features(self, env):
        """Get features of the termination action"""
        pres_node_map = env.present_trial.node_map
        term_features = get_normalized_feature_values(
            pres_node_map[0].compute_termination_feature_values(
                self.features, adaptive_satisficing={}
            ),
            self.features,
            self.normalized_features,
        )
        return term_features

    def get_best_paths_expectation(self, env):
        if len(self.previous_best_paths) == 0:
            return 0
        else:
            trial = env.present_trial
            node_map = trial.node_map
            path_values = []
            for path in self.previous_best_paths:
                path_value = 0
                for node in path:
                    if node_map[node].observed:
                        path_value += node_map[node].value
                    else:
                        path_value += node_map[node].expected_value
                path_values.append(path_value)
            return np.max(path_values)
            # This reflects the fact that we're taking
            # only the best of the best paths into consideration

    def store_best_paths(self, env):
        branch_map = env.present_trial.branch_map
        trial = env.present_trial
        node_map = trial.node_map
        path_sums = {}
        for branch in range(1, len(branch_map) + 1):
            total_sum = 0
            for node in branch_map[branch]:
                if node_map[node].observed:
                    total_sum += node_map[node].value
                else:
                    if node != 0:
                        total_sum += node_map[node].expected_value
            path_sums[branch] = total_sum
        max_path_sum = max(path_sums.values())
        best_paths = [
            branch_map[k][1:] for k, v in path_sums.items() if v == max_path_sum
        ]
        self.previous_best_paths = best_paths

    def get_pseudo_reward(self, env):
        pr = 0
        if self.use_pseudo_rewards:
            comp_value = self.get_best_paths_expectation(env)
            mer = self.get_term_reward(env)
            pr = self.pr_weight * (mer - comp_value)
        return pr

    def run_multiple_simulations(
        self, env, num_simulations, compute_likelihood=False, participant=None
    ):
        env.attach_features(self.features, self.normalized_features)
        env.reset()
        if compute_likelihood and not participant:
            raise ValueError(
                "Likelihood can only be computed for a participant's actions"
            )
        simulations_data = defaultdict(list)
        for _ in range(num_simulations):
            trials_data = self.simulate(
                env, compute_likelihood=compute_likelihood, participant=participant
            )
            for param in ["r", "w", "a", "loss", "decision_params", "s", "info"]:
                if param in trials_data:
                    simulations_data[param].append(trials_data[param])
            # reset participant, needed for likelihood object fxn
            participant.reset()
        total_m_mers = []
        for i in range(len(simulations_data["a"])):
            m_mers = get_termination_mers(
                env.ground_truth, simulations_data["a"][i], env.pipeline
            )
            total_m_mers.append(m_mers)
        simulations_data["mer"] = total_m_mers
        return simulations_data
