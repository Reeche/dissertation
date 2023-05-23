from collections import defaultdict
import torch
import mpmath as mp
import numpy as np
from pyro.distributions import DirichletMultinomial

from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.utils.learning_utils import (break_ties_random,
                                              estimate_bayes_glm,
                                              get_log_norm_cdf,
                                              get_log_norm_pdf, norm_integrate,
                                              rows_mean, sample_coeffs)


class ModelBased(Learner):
    """Base class of the Model-based model"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.init_model_params()

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        # node distributions follow a dirichlet distribution with two alpha parameters (basically a beta)
        self.dirichlet_alpha = {key: (1, 1) for key in range(13)}
        # click likelihood is a multinomial distribution
        self.multinomial_k = range(-48, 49)  # todo: what if the range is not known
        self.multinomial_n = 1
        self.action_log_probs = []


    def init_distributions(self, env):
        num_available_actions = len(env.get_available_actions())
        self.action_distributions = {}
        for i in range(num_available_actions):
            self.action_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor([self.dirichlet_alpha[i][0], self.dirichlet_alpha[i][1]]))

    def update_params(self, env):
        if self.is_null:
            return
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        num_available_actions = len(env.get_available_actions())

        # need the observed value of the node and not reward (which is the -1 click cost)
        observed_value = env.ground_truth[env.present_trial_number][env.observed_action_list[env.present_trial_number]]

        if observed_value > 24:  # alpha + 1 todo: need to choose sensible threshold
            self.dirichlet_alpha[env.observed_action_list[env.present_trial_number]][0] = + 1
        else:  # beta + 1
            self.dirichlet_alpha[env.observed_action_list[env.present_trial_number]][1] = + 1

        for i in range(num_available_actions):
            self.action_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor([self.dirichlet_alpha[i][0], self.dirichlet_alpha[i][1]]),
                total_count=env.present_trial_number + 1) # because present_trial_number start at 0

    def get_action(self, trial_info):
        """Get the best action and its features in the given state"""
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
        else:
            ## get the best action according to the expected return of the dirichlet/dirichlet_multinomial distributions
            expected_return = [action_distribution.mean for action_distribution in self.action_distributions]

            ## chose the action with the highest expected_return
            best_action = expected_return.index(max(expected_return))

            action = best_action
        return action

    def perform_action_updates(self, env):
        #if self.use_pseudo_rewards:
        #    self.pseudo_reward = self.get_pseudo_reward(env)
        #value_estimate = reward - self.subjective_cost + self.pseudo_reward
        self.update_params(env)

    def perform_end_episode_updates(self, env, reward, taken_path):
        selected_action = 0
        delay = env.get_feedback({"taken_path": taken_path, "action": selected_action})
        pr = self.get_pseudo_reward(env)
        value_estimate = reward + pr - self.delay_scale * delay
        self.update_params(env)
        self.update_rewards.append(
            value_estimate
        )

    def take_action(self, env, trial_info):
        action = self.get_action(trial_info)
        if self.compute_likelihood:
            pi = trial_info["participant"]
            # todo: the likelihood that the model would have taken the same action as the pid, i.e. likelihood of pid action
            self.action_log_probs.append(self.action_distributions[action]) #todo: check this
            s_next, r, done, _ = env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
            s_next, reward, done, taken_path = env.step(action)
        return s_next, action, reward, done, taken_path

    def act_and_learn(self, env, trial_info=None):
        if trial_info is None:
            trial_info = {}

        term_reward = self.get_term_reward(env)
        term_features = self.get_term_features(env)
        self.term_rewards.append(term_reward)
        self.store_best_paths(env)
        (
            s_next,
            action,
            reward,
            done,
            taken_path,
            delay,
            features,
        ) = self.take_action(env, trial_info)
        if not done:
            a_next = self.get_action(env, trial_info)
            self.perform_action_updates(
                env,
                reward - self.delay_scale * delay,
                term_features,
                term_reward,
                features,
            )
        else:  # if done; hierarchical models never enter here
            self.update_features.append(features)
            self.perform_end_episode_updates(env, features, reward, taken_path)
        return action, reward, done, taken_path


    def simulate(self, env, compute_likelihood=False, participant=None):
        self.init_model_params()
        self.compute_likelihood = compute_likelihood
        env.reset()
        trials_data = defaultdict(list)
        num_trials = env.num_trials

        for trial_num in range(num_trials):
            self.previous_best_paths = []
            self.num_actions = len(env.get_available_actions())
            self.update_rewards  = []
            actions, rewards, self.term_rewards = [], [], []
            done = False
            while not done:
                action, reward, done, taken_path = self.act_and_learn(
                    env, trial_info={"participant": participant}
                )
                rewards.append(reward)
                actions.append(action)
                if done:
                    trials_data["taken_paths"].append(taken_path)

            trials_data["r"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(rewards)
            env.get_next_trial()
        # add trial ground truths
        trials_data["envs"] = env.ground_truth
        # Likelihoods are stored in action_log_probs
        if self.action_log_probs:
            trials_data["loss"] = -np.sum(self.action_log_probs)
        else:
            trials_data["loss"] = None
        return dict(trials_data)
