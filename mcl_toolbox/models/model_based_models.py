from collections import defaultdict
import torch
import mpmath as mp
import numpy as np

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
        self.dirichlet_alpha = {key: (1, 1) for key in range(13)}
        self.multinomial_k = range(-48, 49) #todo: what if the range is not known
        self.multinomial_n = 1

    def init_distributions(self, env):
        available_actions = env.get_available_actions()
        num_available_actions = len(available_actions)
        self.action_distributions = {}
        for i in range(num_available_actions):
            self.action_distributions[i] = torch.distributions.Dirichlet(torch.tensor([self.dirichlet_alpha[0][0], self.dirichlet_alpha[0][1]]))

    def update_params(self, r):
        if self.is_null:
            return
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        #todo: need to write down the equations for the update



    def get_action(self, env, trial_info):
        """Get the best action and its features in the given state"""
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
        else:
            ## get the best action according to the expected return of the dirichlet distributions
            expected_return = [action_distribution.mean for action_distribution in self.action_distributions]

            ## chose the action with the highest expected_return
            best_action = expected_return.index(max(expected_return))

            action = best_action
        return action

    def perform_action_updates(self, env, reward):
        if self.use_pseudo_rewards:
            self.pseudo_reward = self.get_pseudo_reward(env)
        value_estimate = reward - self.subjective_cost + self.pseudo_reward
        self.update_params(value_estimate)


    def perform_end_episode_updates(self, env, features, reward, taken_path):
        selected_action = 0
        delay = env.get_feedback({"taken_path": taken_path, "action": selected_action})
        pr = self.get_pseudo_reward(env)
        value_estimate = reward + pr - self.delay_scale * delay
        self.update_params(value_estimate)
        self.update_rewards.append(
            value_estimate
        )


    def store_action_likelihood(self, env, given_action):
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
        action_index = available_actions.index(given_action)
        num_available_actions = len(available_actions)
        feature_vals = self.get_action_features(env)
        dists = np.zeros((num_available_actions, 2))
        cov = np.linalg.inv(self.precision*self.num_samples)
        for index, action in enumerate(available_actions):
            computed_features = feature_vals[action]
            dists[index][0] = np.dot(computed_features, self.mean)
            dists[index][1] = np.dot(
                np.dot(computed_features, cov), computed_features.T
            )

        means = dists[:, 0]
        sigmas = np.sqrt(dists[:, 1])
        # Very important to select good bounds for proper sampling.
        ub = np.max(means + 5 * sigmas)
        lb = np.min(means - 5 * sigmas)
        if num_available_actions == 1:
            selected_action_prob = 1
        else:
            # samples = np.random.multivariate_normal(means, np.diag(variances), size=n_samples)
            # argmaxes = np.argmax(samples, axis=1)
            # counts = Counter(argmaxes)
            # get the probability of this certain action
            # integrate normal pdf(action, mean, sigma) using lb, ub as bounds
            selected_action_prob = mp.quad(
                lambda x: norm_integrate(x, action_index, means, sigmas),
                [lb, ub],
                maxdegree=10,
            )
            #integrate over predicted q-value of the action
        eps = self.eps
        log_prob = float(
            str(
                mp.log(
                    (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
                )
            )
        )
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]

    def take_action(self, env, trial_info):
        action, features = self.get_action_details(env, trial_info)
        delay = env.get_feedback({"action": action})
        if self.compute_likelihood:
            pi = trial_info["participant"]
            self.store_action_likelihood(env, action) #only click action and not the termination action
            s_next, r, done, _ = env.step(action)
            reward, taken_path, done = pi.make_click()
            # assert r == reward #doesn't make sense because the path is not the same
        else:
            s_next, reward, done, taken_path = env.step(action)
        return s_next, action, reward, done, taken_path, delay, features

    def act_and_learn(self, env, trial_info=None):
        if trial_info is None:
            trial_info = {}
        end_episode = False
        if "end_episode" in trial_info:
            end_episode = trial_info["end_episode"]
        if not end_episode:
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
                a_next, next_features = self.get_action_details(env, trial_info)
                self.update_features.append(features)
                self.perform_action_updates(
                    env,
                    next_features,
                    reward - self.delay_scale * delay,
                    term_features,
                    term_reward,
                    features,
                    )
            else: #if done; hierarchical models never enter here
                self.update_features.append(features)
                self.perform_end_episode_updates(env, features, reward, taken_path)
            return action, reward, done, taken_path
        else:  # for hierarchial model, if it says to terminate
            # Should this model learn from the termination action?
            reward = 0
            taken_path = None
            if self.compute_likelihood:
                reward, taken_path, done = trial_info["participant"].make_click()

            # todo: previously, there was no update here. does it make sense that there is no update here?
            # todo: does it make sense to use the termination_features?
            self.perform_end_episode_updates(env, self.get_term_features(env), reward, taken_path)
            return 0, reward, True, taken_path

    def simulate(self, env, compute_likelihood=False, participant=None):
        self.init_model_params()
        self.compute_likelihood = compute_likelihood
        env.reset()
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        if compute_likelihood:
            get_log_norm_pdf.cache_clear()
            get_log_norm_cdf.cache_clear()
        for trial_num in range(num_trials):
            self.previous_best_paths = []
            self.num_actions = len(env.get_available_actions())
            trials_data["w"].append(self.get_current_weights())
            self.update_rewards, self.update_features = [], []
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
