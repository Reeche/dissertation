import gym
import numpy as np
from gym import spaces
import torch
from pydantic import NonNegativeFloat
from mcl_toolbox.env.modified_mouselab import TrialSequence, reward_val
from mouselab.distributions import PointMass, cmax, expectation, sample, smax, Categorical
from mcl_toolbox.utils.env_utils import get_num_actions
from mcl_toolbox.utils.sequence_utils import compute_current_features

ZERO = PointMass(0)


class GenericMouselabEnv(gym.Env):
    """
    This class is the gym environment for the feature based version
    of the Mouselab-MDP. The environment structure is assumed to be a
    symmetric tree
    """

    def __init__(
            self,
            tree,
            init,
            num_trials=1,
            pipeline={"0": ([3, 1, 2], reward_val)},
            ground_truth=None,
            cost=1,
            render_path="mouselab_renders",
            feedback="none",
            q_fn=None,
    ):
        super(GenericMouselabEnv, self).__init__()
        self.pipeline = pipeline
        self.ground_truth = ground_truth
        self.num_trials = num_trials
        self.render_path = render_path
        if isinstance(cost, list):
            cost_weight, depth_weight = cost
            self.cost = lambda depth: -(1 * cost_weight + depth * depth_weight)
            self.repeat_cost = -float("inf")
        elif hasattr(cost, '__call__'):  # if it is a function
            self.cost = cost
            self.repeat_cost = -float("inf")
        else:  # should be a scalar
            self.cost = lambda depth: -(1 * cost)
            self.repeat_cost = -cost * 10
        self.feedback = feedback
        self.q_fn = q_fn
        self.feature_state = None
        self.features = None
        self.normalized_features = None
        if self.feedback == "meta" and self.q_fn is None:
            raise ValueError("Q-function is required to compute metacognitive feedback")
        self.construct_env()
        self.tree = tree
        self.init = init
        self.term_action = 0

    def custom_same_env_init(self, env, num_trials):
        self.num_trials = num_trials
        ground_truths = [env] * self.num_trials
        self.ground_truth = ground_truths
        self.construct_env()

    def participant_init(self, ground_truth):
        self.num_trials = len(ground_truth)
        self.ground_truth = ground_truth
        self.construct_env()

    def construct_env(self):
        self.trial_sequence = TrialSequence(
            self.num_trials, self.pipeline, self.ground_truth
        )
        self.present_trial_num = 0
        self.trial_init()
        if not self.ground_truth:
            self.ground_truth = self.trial_sequence.ground_truth

    def trial_init(self):
        trial_num = self.present_trial_num
        self.num_nodes = len(self.trial_sequence.trial_sequence[trial_num].node_map)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(self.num_nodes,), dtype=np.float64
        )
        self.present_trial = self.trial_sequence.trial_sequence[trial_num]
        reward_function = self.pipeline[self.present_trial_num][1]
        self.node_distribution = [[0]] + [
            reward_function(d) for d in range(1, self.present_trial.max_depth + 1)
        ]
        self._compute_expected_values()
        self._construct_state()
        self.observed_action_list = []
        self.num_actions = len(self.get_available_actions())

    def _construct_state(self):
        self._state = [0] + [
            self.node_distribution[self.present_trial.node_map[node_num].depth]
            for node_num in range(1, self.num_nodes)
        ]

    def _compute_expected_values(self):
        self.expected_values = [0] + [
            self.node_distribution[
                self.present_trial.node_map[node_num].depth
            ].expectation()
            for node_num in range(1, self.num_nodes)
        ]

    def get_next_trial(self):
        if self.present_trial_num == self.num_trials - 1:
            return -1
        self.present_trial_num += 1
        self.trial_init()
        self.observed_action_list = []
        if self.features is not None:
            self.feature_state = self.construct_feature_state()
        return None

    def reset_trial(self):
        self._compute_expected_values()
        self._construct_state()
        self.present_trial.reset_observations()
        self.observed_action_list = []
        if self.features is not None:
            self.feature_state = self.construct_feature_state()

    def reset(self):
        self.construct_env()
        if self.features is not None:
            self.feature_state = self.construct_feature_state()
        return self._state

    def step(self, action):
        info = {}
        reward = self.cost(self.present_trial.node_map[action].depth)
        done = False
        if action in self.observed_action_list:
            return self._state, self.repeat_cost, False, {}
        self.observed_action_list.append(action)
        node_map = self.present_trial.node_map
        if not action == 0:
            self.present_trial.node_map[action].observe()
        else:
            done = True
            best_expected_path = self.present_trial.get_best_expected_path()
            info = best_expected_path[1:]
            reward = 0
            for node in best_expected_path:
                reward += self.present_trial.node_map[node].value
                # self.present_trial.node_map[node].observe()
        self._state[action] = node_map[action].value
        if self.features is not None:
            self.feature_state = self.construct_feature_state()
        return self._state, reward, done, info

    def render(self, dir_path=None):
        pass

    def get_random_env(self):
        trial_sequence = TrialSequence(num_trials=1, pipeline=self.pipeline)
        return trial_sequence.ground_truth[0]

    def get_ground_truth(self):
        return self.ground_truth

    def get_available_actions(self):
        nodes = [n.label for n in self.present_trial.unobserved_nodes]
        return nodes

    def get_best_paths(self):
        trial = self.present_trial
        expected_path_values = trial.get_path_expected_values()
        best_paths = [
            k
            for k, v in expected_path_values.items()
            if v == max(expected_path_values.values())
        ]
        return set(best_paths)

    def get_action_feedback(self, taken_path):
        delay = self.present_trial.get_action_feedback(taken_path)
        return delay

    def get_metacognitive_feedback(self, action):
        present_state = self.get_state()
        available_actions = self.get_available_actions()
        qs = []
        for a in available_actions:
            qs.append(self.q_fn[(present_state, self.env_action(a))])
        max_q = max(qs)
        mcfb_delay = 2 + max_q - self.q_fn[(present_state, self.env_action(action))]
        if mcfb_delay == 2:
            mcfb_delay = 0
        return mcfb_delay

    def get_feedback(self, info):
        if self.feedback == "action":
            if "taken_path" in info:
                return self.get_action_feedback(info["taken_path"])
        elif self.feedback == "meta":
            return self.get_metacognitive_feedback(info["action"])
        return 0

    def get_state(self):
        rd = self.pipeline[self.present_trial_num][1]
        branching = self.pipeline[self.present_trial_num][0]
        num_nodes = get_num_actions(branching)
        state = [0]
        for node_num in range(num_nodes):
            node = self.present_trial.node_map[node_num]
            if node_num != 0:
                if node.observed:
                    state.append(self.present_trial.ground_truth[node_num])
                else:
                    # take care of transfer task
                    dist = rd(node.depth)
                    state.append(Categorical(dist.vals, dist.probs))
        return tuple(state)

    def construct_feature_state(self):
        self.feature_state = compute_current_features(
            self.present_trial, self.features, self.normalized_features
        )
        return self.feature_state

    def get_feature_state(self):
        if self.feature_state is None:
            if self.features is not None:
                return self.construct_feature_state()
            else:
                # see base_learner.py for example of attaching features
                raise ValueError("Features need to be attached with attach_features method.")
        return self.feature_state

    def env_action(self, a):
        branching = self.pipeline[0][0]
        num_nodes = get_num_actions(branching)
        if a == 0:
            return num_nodes
        else:
            return a

    def get_term_reward(self):
        """Get the max expected return in the current state"""
        pres_node_map = self.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    # How would you run say transfer task easily?
    def attach_features(self, features, normalized_features):
        self.features = features
        self.normalized_features = normalized_features

    def expectation_dirichlet(self, distribution, value_range):
        """
        Args:
            distribution: Dirichlet distribution of selected node
            value_range: range of possible values for the node

        Returns: expected value of the node given observations (alphas)

        """
        # expectation of the selected node
        # value range to dict with index as key
        value_dict = {item: index for index, item in enumerate(value_range)}
        if hasattr(distribution, "concentration"):
            expectation = 0
            for value in value_range:
                idx = value_dict[value]
                prob = distribution.concentration[idx] / torch.sum(distribution.concentration)
                expectation += prob * value
            return expectation
        else:
            return 0

    # def node_value(self, node, value_range, state=None):
    #     """A distribution over total rewards after the given node."""
    #     state = state if state is not None else self._state
    #     return max(
    #         (self.node_value(n1, state) + self.expectation_dirichlet(state[n1], value_range) for n1 in self.tree[node]),
    #         # (self.node_value(n1, state) + self.state_expectations[n1] for n1 in self.tree[node]),
    #         default=ZERO
    #     )
    #
    # def term_reward(self, value_range, state=None):
    #     """A distribution over the return gained by acting given a belief state."""
    #     state = state if state is not None else self._state
    #     return self.node_value(0, state, value_range)
    #
    # def expected_term_reward(self, state, value_range, action=None):
    #     expected_term_reward = self.term_reward(state, value_range).expectation()
    #     return np.sign(expected_term_reward) * np.abs(expected_term_reward)

    # def node_value_after_observe(self, obs, node, state):
    #     """A distribution over the expected value of node, after making an observation.
    #
    #     obs can be a single node, a list of nodes, or 'all'
    #     """
    #     obs_tree = self.to_obs_tree(state, node, obs)
    #     return node_value_after_observe(obs_tree)
    #
    # def myopic_voc(self, action, state) -> NonNegativeFloat:
    #     self.state_expectations = [self.expectation_dirichlet(state[node]) for node in range(len(state))]
    #
    #     if action == self.term_action:
    #         return 0
    #     else:
    #         gain_from_inspecting = self.node_value_after_observe(
    #             (action,), 0, state
    #         ).expectation()
    #
    #         corrected_gain_from_inspecting = np.sign(gain_from_inspecting) * np.abs(
    #             gain_from_inspecting)
    #         return corrected_gain_from_inspecting - self.expected_term_reward(state)
    #         # return corrected_gain_from_inspecting - self.get_term_reward()

    def to_obs_tree(self, state, node, action, value_range, sort=True):
        """

        Args:
            state: the current state of the mouselab MDP, i.e. value and their probabilities
            node: 0
            action:
            sort:

        Returns:

        """
        maybe_sort = sorted if sort else lambda x: x

        def rec(n):
            # if n in action: #if n in action, it always know the actual ground truth value and know exactly where to click
            if n in self.observed_action_list: #if observed_action_list, the myopic values in the beginning always resets, WHY?
                subjective_reward = torch.tensor(self.present_trial.ground_truth[n])
            else:
                # start at node 0, i.e. state[0], i.e. dirichlet distribution of node 0, which is 0 in the beginning
                subjective_reward = self.expectation_dirichlet(state[n], value_range)
                # subjective_reward = self.state_expectations[n]
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            return (subjective_reward, children)

        return rec(node)

    def node_value_after_observe(self, state, node, action, value_range):
        """A distribution over the expected value of node, after making an observation.

        obs can be a single node, a list of nodes, or 'all'
        """
        # obs_tree returns a tree with expected value of each node
        obs_tree = self.to_obs_tree(state, node, action, value_range)
        # if self.exact:
        #     return exact_node_value_after_observe(obs_tree)
        # else:
        # print("tree", obs_tree)
        return node_value_after_observe(obs_tree)

    def myopic_voc(self, action, state, value_range) -> NonNegativeFloat:
        # self.state_expectations = [self.expectation_dirichlet(state[node], value_range) for node in range(len(state))]

        # loops through all actions, not necessarily the ones that have been observed
        if action == self.term_action:
            # return 0
            return self.get_term_reward()
        else:
            # gain_from_inspecting = self.node_value_after_observe(action, 0, state).expectation()
            gain_from_inspecting = self.node_value_after_observe(state=state, node=0, action=(action,),
                                                                 value_range=value_range).expectation()
            # print("gain", gain_from_inspecting)
            corrected_gain_from_inspecting = np.sign(gain_from_inspecting) * np.abs(gain_from_inspecting)
            return corrected_gain_from_inspecting - self.get_term_reward()
            # return corrected_gain_from_inspecting - self.expected_term_reward(state, value_range)


def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return cmax(children, default=ZERO)


def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(exact_node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return cmax(children, default=ZERO)


# class ModStateGenericMouselabEnv(GenericMouselabEnv):
#     def __init__(
#             self,
#             num_trials=1,
#             pipeline={"0": ([3, 1, 2], reward_val)},
#             ground_truth=None,
#             cost=1,
#             render_path="mouselab_renders",
#     ):
#         super().__init__(num_trials, pipeline, ground_truth, cost, render_path)
#
#     def _construct_state(self):
#         self._state = np.array([0] + [0 for node_num in range(1, self.num_nodes)])
#
#     def step(self, action):
#         S, reward, done, info = super().step(action)
#         for node in self.present_trial.node_map.keys():
#             if node not in self.observed_action_list:
#                 S[node] = 0
#         S = np.array(S)
#         return S, reward, done, info


class DummyParticipant:
    """Creates a participant object which contains all details about the participant

    Returns:
        Participant -- Contains details such as envs, scores, clicks, taken paths,
                       strategies and weights at each trial.
    """

    def __init__(self, pipeline, num_trials):
        self.all_trials_data = self.get_all_trials_data()
        self.num_trials = num_trials
        self.pipeline = pipeline

    @property  # This is done on purpose to induce stochasticity
    def envs(self):
        envs = GenericMouselabEnv(self.num_trials, self.pipeline).ground_truth
        self.trial_envs = envs
        return envs

    def get_envs(self):
        return self.trial_envs

    def get_all_trials_data(self):
        total_data = {
            "actions": {},
            "rewards": {},
            "taken_paths": {},
            "strategies": {},
            "temperature": {},
        }
        return total_data


class DummyParticipantNew:
    """Creates a participant object which contains all details about the participant

    Returns:
        Participant -- Contains details such as envs, scores, clicks, taken paths,
                       strategies and weights at each trial.
    """

    def __init__(self, pipeline, num_trials):
        self.num_trials = num_trials
        self.pipeline = pipeline
        self.envs = GenericMouselabEnv(self.num_trials, self.pipeline)
        self.clicks = []
        self.strategies = []
        self.scores = []
        self.weights = []
        self.all_trials_data = self.get_all_trials_data()

    def get_all_trials_data(self):
        total_data = {
            "actions": {},
            "rewards": {},
            "taken_paths": {},
            "strategies": {},
            "temperature": {},
        }
        return total_data
