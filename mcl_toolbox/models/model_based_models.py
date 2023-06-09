import random
from collections import defaultdict
import torch
from torch.autograd import Variable
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
        self._init_model_params()
        self._init_distributions()

    def _init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        # each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        dirichlet_alpha = {key: 1 for key in self.value_range}
        self.dirichlet_alpha_dict = {}
        # alpha need to be n x m, e.g. 13 x range
        for i in range(self.num_available_nodes):
            self.dirichlet_alpha_dict[i] = dirichlet_alpha

    def _init_distributions(self):
        self.node_distributions = {}
        # create tensor of length value_range

        # create node distribution for all nodes
        for i in range(self.num_available_nodes):
            self.node_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor(list(self.dirichlet_alpha_dict[i].values())))

    def update_params(self, reward, action):
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)

        # need the observed value of the node minus the reward (which is the -1 click cost)
        # todo: check if observed_action_list is correct
        observed_value = self.env.ground_truth[self.env.present_trial_num][self.env.observed_action_list[0]] - reward
        self.dirichlet_alpha_dict[action - 1][
            int(observed_value)] += 1  # -1 because dirichlet start at 0 but 0  is termination action and therefore action start 1

        for i in range(self.num_available_nodes):
            self.node_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor((list(self.dirichlet_alpha_dict[i].values()))),
                total_count=self.env.present_trial_num + 1)  # because present_trial_number start at 0

    def get_myopic_action(self) -> int:
        ## chose the action with the highest expected_return (myopic policy)
        max_value = float('-inf')  # Initialize with negative infinity
        max_index = None

        for index, sublist in enumerate(self.expected_return()):
            current_max = max(sublist)
            if current_max > max_value:
                max_value = current_max
                max_index = index

        if random.random() < self.inverse_temp:
            return random.randint(0, self.num_available_nodes)
        else:
            return max_index

    def expected_return(self):
        ## get the best action according to the expected return of the dirichlet/dirichlet_multinomial distributions
        expected_return = []
        for _, node_distribution in self.node_distributions.items():
            expected_return.append(node_distribution.mean)
        return expected_return

    def get_pseudo_reward(self):
        pr = 0
        # maximum expected reward from previous belief state
        comp_value = self.get_best_paths_expectation(self.env)
        # maximum expected reward from current belief state (if terminate now)
        mer = self.get_term_reward(self.env)
        pr = self.pr_weight * (mer - comp_value)
        return pr

    def perform_updates(self, reward, action):
        # if self.pseudo_reward:
        #     self.pseudo_reward = self.get_pseudo_reward()
        value_estimate = reward  # + self.pseudo_reward
        self.update_params(value_estimate, action)

    # def perform_end_episode_updates(self, reward, action, taken_path): #todo: if not likelihood is used as criteria
    #     selected_action = 0
    #     # delay = self.env.get_feedback({"taken_path": taken_path, "action": selected_action}) #todo: what is delay?
    #     # pr = self.get_pseudo_reward(self.env)
    #     # value_estimate = reward + pr - self.delay_scale * delay
    #     value_estimate = reward
    #     self.update_params(value_estimate, action)

    def calculate_likelihood(self):
        # for expected return of each action, choose the largest ones, which is the action score
        action_score = []
        for index, sublist in enumerate(self.expected_return()):
            action_score.append(max(sublist))
        # action_score = self.inverse_temp * np.array(action_score)

        softmax_vals = F.log_softmax(torch.tensor(action_score), dim=0)
        softmax_vals = torch.exp(softmax_vals)
        return softmax_vals / softmax_vals.sum()

    def take_action(self, trial_info):
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
            action_likelihood = self.calculate_likelihood()  # todo: how to determine prob of termination?
            if action != 0:
                action = action - 1
                # how likely model would have taken the action
                self.action_log_probs.append(action_likelihood[action])

        # what would the model have done and received as reward?
        best_action = self.get_myopic_action()
        _, reward, d, tp = self.env.step(best_action)
        # need this to get the done, taken_path information and update # of clicks and trials
        _, taken_path, done = pi.make_click()
        return action, reward, done, taken_path

    def act_and_learn(self, trial_info=None):
        if trial_info is None:
            trial_info = {}
        action, reward, done, taken_path = self.take_action(trial_info)
        if action != 0:
            self.perform_updates(reward, action)  # todo: check that reward is the click cost and the termination reward
        # else:
        #     self.perform_end_episode_updates(reward, action, taken_path)
        return action, reward, done, taken_path

    def simulate(self, params):
        self.inverse_temp = params['inverse_temp']
        self._init_model_params()

        self.env.reset()
        self.participant_obj.reset()
        trials_data = defaultdict(list)
        num_trials = self.env.num_trials
        for trial_num in range(num_trials):
            actions, rewards = [], []
            done = False
            while not done:
                action, reward, done, taken_path = self.act_and_learn(trial_info={"participant": self.participant_obj})
                rewards.append(reward)
                actions.append(action)
                if done:
                    trials_data["taken_paths"].append(taken_path)

            trials_data["r"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(rewards)
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
