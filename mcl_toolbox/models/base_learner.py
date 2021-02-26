from abc import ABC, abstractmethod
from collections import defaultdict

from mcl_toolbox.env.modified_mouselab import get_termination_mers
from mcl_toolbox.utils.learning_utils import get_normalized_feature_values


class Learner(ABC):
    """Base class of RL models implemented for the Mouselab-MDP paradigm."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def simulate(self, env):
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
            pres_node_map[0].compute_termination_feature_values(self.features, adaptive_satisficing={}), 
            self.features, self.normalized_features)
        return term_features

    def run_multiple_simulations(self, env, num_simulations, compute_likelihood=False, participant=None):
        env.reset()
        if compute_likelihood and not participant:
            raise ValueError("Likelihood can only be computed for a participant's actions")
        simulations_data = defaultdict(list)
        for _ in range(num_simulations):
            trials_data = self.simulate(env, compute_likelihood=compute_likelihood, participant=participant)
            for param in ['r', 'w', 'a', 'loss', 'decision_params', 's', 'info']:
                if param in trials_data:
                    simulations_data[param].append(trials_data[param])
        total_m_mers = []
        for i in range(len(simulations_data['a'])):
            m_mers = get_termination_mers(env.ground_truth, simulations_data['a'][i], env.pipeline)
            total_m_mers.append(m_mers)
        simulations_data['mer'] = total_m_mers
        return simulations_data