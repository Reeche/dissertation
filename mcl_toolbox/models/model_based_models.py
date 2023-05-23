from collections import defaultdict
import torch
import numpy as np
from pyro.distributions import DirichletMultinomial

from mcl_toolbox.models.base_learner import Learner


class ModelBased(Learner):
    """Base class of the Model-based model"""

    def __init__(self, participant, env, value_range):
        self.participant = participant
        self.env = env
        self.value_range = value_range
        self._init_model_params()
        self._init_distributions()


    def _init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        # each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        dirichlet_alpha = {key: 1 for key in self.value_range}
        self.dirichlet_alpha_dict = {}
        # alpha need to be n x m, e.g. 13 x range
        num_available_actions = len(self.env.get_available_actions())
        for i in range(num_available_actions):
            self.dirichlet_alpha_dict[i] = dirichlet_alpha
        # click likelihood is a multinomial distribution #todo: it is not really used anywhere since we already have the posterior
        #self.multinomial_k = self.value_range
        #self.multinomial_n = 1
        self.action_log_probs = []

    def _init_distributions(self):
        num_available_actions = len(self.env.get_available_actions())
        self.node_distributions = {}
        # create tensor of length value_range

        # create node distribution for all nodes
        for i in range(num_available_actions):
            self.node_distributions[i] = torch.distributions.Dirichlet(torch.tensor(list(self.dirichlet_alpha_dict[i].values())))

    def update_params(self, reward, action):
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        num_available_actions = len(self.env.get_available_actions())

        # need the observed value of the node minus the reward (which is the -1 click cost)
        #todo: check if observed_action_list is correct
        observed_value = self.env.ground_truth[self.env.present_trial_num][self.env.observed_action_list[0]] - reward
        self.dirichlet_alpha_dict[action][int(observed_value)] += 1 #todo: for the chosen action

        for i in range(num_available_actions):
            self.node_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor((list(self.dirichlet_alpha_dict[i].values()))),
                total_count=self.env.present_trial_num + 1)  # because present_trial_number start at 0

    @staticmethod
    def find_index_of_list_with_largest_value(list_of_lists):
        max_value = float('-inf')  # Initialize with negative infinity
        max_index = None

        for index, sublist in enumerate(list_of_lists):
            current_max = max(sublist) # todo: do we need to break ties, i.e. randomly choose if there are ties? Otherwise max always takes the first action
            if current_max > max_value:
                max_value = current_max
                max_index = index

        return max_index

    def get_action(self, trial_info):
        """Get the best action and its features in the given state"""
        expected_return = []
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
        else:
            ## get the best action according to the expected return of the dirichlet/dirichlet_multinomial distributions
            for _, node_distribution in self.node_distributions.items():
                expected_return.append(node_distribution.mean)
            ## chose the action with the highest expected_return
            best_action = self.find_index_of_list_with_largest_value(expected_return)
            action = best_action
        return action

    def perform_updates(self, reward, action):
        # if self.use_pseudo_rewards:
        #     self.pseudo_reward = self.get_pseudo_reward(self.env)
        value_estimate = reward #- self.subjective_cost #+ self.pseudo_reward
        self.update_params(value_estimate, action)

    def perform_end_episode_updates(self, reward, action, taken_path):
        selected_action = 0
        # delay = self.env.get_feedback({"taken_path": taken_path, "action": selected_action}) #todo: what is delay?
        #pr = self.get_pseudo_reward(self.env)
        # value_estimate = reward + pr - self.delay_scale * delay
        value_estimate = reward
        self.update_params(value_estimate, action)

    def take_action(self, trial_info):
        action = self.get_action(trial_info)
        if self.compute_likelihood:
            pi = trial_info["participant"]
            # todo: the likelihood that the model would have taken the same action as the pid, i.e. likelihood of pid action
            # self.action_log_probs.append(self.node_distributions[pi.current_click].mean[???])  # todo: check this
            _, r, done, _ = self.env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
            _, reward, done, taken_path = self.env.step(action)
        return action, reward, done, taken_path

    def act_and_learn(self, trial_info=None):
        if trial_info is None:
            trial_info = {}
        done = False
        while not done:
            action, reward, done, taken_path = self.take_action(trial_info)
            self.perform_updates(reward, action) #todo: check that reward is the click cost and the termination reward
        else:
            self.perform_end_episode_updates(reward, action, taken_path)
        return action, reward, done, taken_path

    def simulate(self, compute_likelihood=False, participant=None):
        self._init_model_params()
        self.compute_likelihood = compute_likelihood
        self.env.reset()
        trials_data = defaultdict(list)
        num_trials = self.env.num_trials

        for trial_num in range(num_trials):
            self.num_actions = len(self.env.get_available_actions())
            actions, rewards = [], []
            done = False
            while not done:
                action, reward, done, taken_path = self.act_and_learn(trial_info={"participant": participant})
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
        return dict(trials_data)
