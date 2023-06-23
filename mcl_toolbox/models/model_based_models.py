import random
from collections import defaultdict
import torch
import math
import torch.nn.functional as F
import numpy as np
from pyro.distributions import DirichletMultinomial
from mcl_toolbox.models.base_learner import Learner
from hyperopt import STATUS_OK


class ModelBased(Learner):
    """Base class of the Model-based model"""

    def __init__(self, participant, env, value_range, compute_likelihood, participant_obj):
        self.participant = participant
        self.participant_obj = participant_obj
        self.env = env
        self.value_range = value_range
        self.action_log_probs = []
        self.pseudo_rewards = None
        self.compute_likelihood = compute_likelihood
        self.num_available_nodes = len(self.env.get_available_actions()) - 1
        self.init_model_params()
        self.init_distributions()

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        # each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        dirichlet_alpha = {key: 1 for key in self.value_range}
        self.dirichlet_alpha_dict = {}
        # alpha need to be n x m, e.g. 13 x range
        for i in range(1, self.num_available_nodes + 1):
            self.dirichlet_alpha_dict[i] = dirichlet_alpha

    def init_distributions(self):
        self.node_distributions = {}
        # create tensor of length value_range

        # create node distribution for all nodes
        for i in range(1, self.num_available_nodes + 1):
            self.node_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor(list(self.dirichlet_alpha_dict[i].values())))

    def perform_updates(self, action):
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        # observed_value should only be the node value, i.e. model of the env and not model of the cost
        observed_value = self.env.ground_truth[self.env.present_trial_num][action]
        self.dirichlet_alpha_dict[action][int(observed_value)] = self.dirichlet_alpha_dict[action][
                                                                     int(observed_value)] + 1
        for i in range(1, self.num_available_nodes + 1):
            self.node_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor((list(self.dirichlet_alpha_dict[i].values()))),
                total_count=self.env.present_trial_num + 1)  # because present_trial_number start at 0

    def get_best_action(self) -> int:
        ## chose the action with the highest expected_return (myopic policy)
        action_values = self.myopic_values

        if random.random() < self.inverse_temp:
            return random.randint(0, self.num_available_nodes)
        else:
            return action_values.index(max(action_values))

    @property
    def term_reward(self):
        """Get the max expected return in the current state"""
        pres_node_map = self.env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    def mer_for_action(self, action):
        """
        Loops through all branches and get expected value of each branch /path
        Returns: a dict that contains the path and its expected value

        """
        # self.branch_map is a dict containing the branches and indeces,
        # e.g. for 3 step task {1: (0, 1, 2, 3), 2: (0, 1, 2, 4), 3: (0, 5, 6, 7), 4: (0, 5, 6, 8), 5: (0, 9, 10, 11), 6: (0, 9, 10, 12)}
        num_branches = len(self.env.present_trial.branch_map)
        expected_path_values = {}
        for i in range(1, num_branches + 1):
            path = self.env.present_trial.branch_map[i]
            mer = 0
            for node_num in path:
                node = self.env.present_trial.node_map[node_num]  # node object
                if node.observed:
                    mer += node.value
                else:
                    if node_num != 0:  # if not the starting node
                        # if deliberating about clicking the node, what is the expected return (MER)?
                        mer += max(self.node_distributions[action].mean)  # todo: max or average?
                        # MER for other actions are all 0
            expected_path_values[i] = mer
        return max(expected_path_values.values())

    def node_depth(self, action):
        if action in [1, 5, 9]:
            return 1
        elif action in [2, 6, 10]:
            return 2
        elif action in [3, 4, 7, 8, 10, 11]:
            return 3

    @property
    def myopic_values(self) -> list:
        myopic_values = []
        for action in range(1, self.num_available_nodes + 1):
            mer = self.mer_for_action(action)
            myopic_value = mer - self.term_reward - self.env.cost(self.node_depth(action))
            myopic_values.append(myopic_value)
        return myopic_values

    def calculate_likelihood(self):
        # likelihood of termination, value of 0 because no information gain
        termination_value = math.exp(0)
        myopic_values = self.myopic_values
        myopic_values.insert(0, termination_value)
        softmax_vals = F.log_softmax(torch.tensor([x * self.inverse_temp for x in myopic_values]), dim=0)
        softmax_vals = torch.exp(softmax_vals)
        likelihood = softmax_vals / softmax_vals.sum()
        return likelihood

    def take_action(self, trial_info):
        pi = trial_info["participant"]
        if self.compute_likelihood:
            action = pi.get_click()
            action_likelihood = self.calculate_likelihood()
            # how likely model would have taken the action
            self.action_log_probs.append(action_likelihood[action])
            # _, model_reward, _, _ = self.env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
            action = self.get_best_action()
            s_next, reward, done, taken_path = self.env.step(
                action)  # while not done, get the cost; if done bet expected path
            # pid_reward, done, _ = pi.make_click()
        return action, reward, done, taken_path

    def act_and_learn(self, trial_info=None):
        if trial_info is None:
            trial_info = {}
        action, reward, done, taken_path = self.take_action(trial_info)
        if action != 0:
            self.perform_updates(action)
        return action, reward, done, taken_path

    def simulate(self, params):
        self.inverse_temp = params['inverse_temp']
        self.init_model_params()
        self.init_distributions()
        self.env.reset()
        self.participant_obj.reset() #resets number of trial and clicks, nothing else
        trials_data = defaultdict(list)
        num_trials = self.env.num_trials
        for trial_num in range(num_trials):
            actions, rewards = [], []
            done = False
            while not done:
                action, reward, done, taken_path = self.act_and_learn(
                    trial_info={"participant": self.participant_obj})
                rewards.append(reward)
                actions.append(action)
                if done:
                    trials_data["taken_paths"].append(taken_path)

            trials_data["rewards"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(reward)
            self.env.get_next_trial()
        # add trial ground truths
        trials_data["envs"] = self.env.ground_truth
        # Likelihoods are stored in action_log_probs
        if self.action_log_probs:
            trials_data["loss"] = -np.sum(self.action_log_probs)
        else:
            trials_data["loss"] = None
        trials_data["status"] = STATUS_OK
        return dict(trials_data)
