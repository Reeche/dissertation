from collections import defaultdict

import mpmath as mp
import numpy as np

from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.utils.learning_utils import (break_ties_random,
                                              estimate_bayes_glm,
                                              get_log_norm_cdf,
                                              get_log_norm_pdf, norm_integrate,
                                              rows_mean, sample_coeffs)


class LVOC(Learner):
    """Base class of the LVOC model"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.standard_dev = np.exp(params["standard_dev"])
        self.num_samples = int(params["num_samples"])
        self.init_weights = params["priors"]
        self.eps = max(0, min(params["eps"], 1))
        self.no_term = attributes["no_term"]
        self.vicarious_learning = attributes["vicarious_learning"]
        self.termination_value_known = attributes["termination_value_known"]
        self.monte_carlo_updates = attributes["montecarlo_updates"]
        self.init_model_params()
        self.term_rewards = []

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        self.mean = self.init_weights
        self.precision = np.diag([1 / self.standard_dev ** 2] * self.num_features)
        self.gamma_a = 1
        self.gamma_b = 1
        self.action_log_probs = []

    def get_current_weights(self):
        return self.mean.tolist()

    def sample_weights(self):
        """Sample weights from the posterior distribution"""
        sampled_weights = sample_coeffs(
            self.mean, self.precision, self.gamma_a, self.gamma_b, self.num_samples
        )
        return rows_mean(sampled_weights)

    def update_params(self, f, r):
        """Perform Bayesian Regression to update the parameters.

        Arguments:
            f {list} -- Features to be updated against.
            r {float} -- Reward signal against which the parameters are updated.
        """
        if self.is_null:
            return
        self.mean, self.precision, self.gamma_a, self.gamma_b = estimate_bayes_glm(
            f, r, self.mean, self.precision, self.gamma_a, self.gamma_b
        )

    def get_action_features(self, env):
        action_features = env.get_feature_state()
        if self.no_term:
            action_features[0] = np.zeros(self.num_features)
        return np.array(action_features)

    def get_action_details(self, env, trial_info):
        """Get the best action and its features in the given state"""
        feature_vals = self.get_action_features(env)
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
        else:
            available_actions = env.get_available_actions()
            num_available_actions = len(available_actions)

            q = np.zeros(num_available_actions)
            sampled_weights = self.sample_weights()
            for index, action in enumerate(available_actions):
                q[index] = np.dot(sampled_weights, feature_vals[action])

            if self.termination_value_known:
                term_reward = self.get_term_reward(env)
                q[0] = term_reward

            if self.no_term:  # LVOC doesn't terminate (used by the 2-stage model)
                if len(available_actions) != 1:
                    available_actions.remove(0)
                    best_index = break_ties_random(q[1:].tolist())
                else:
                    best_index = break_ties_random(q.tolist())
                best_action = available_actions[best_index]
            else:
                best_index = break_ties_random(q.tolist())
                best_action = available_actions[best_index]

            random_action = np.random.binomial(1, p=self.eps)
            if random_action:
                best_action = np.random.choice(available_actions)
            action = best_action
        features = feature_vals[action]
        return action, features

    def perform_action_updates(
        self, env, next_features, reward, term_features, term_reward, features
    ):
        q = np.dot(self.mean, next_features)
        pr = self.get_pseudo_reward(env)
        self.update_rewards.append(reward + pr - self.subjective_cost)
        self.pseudo_reward = pr
        value_estimate = q + (reward - self.subjective_cost) + pr
        self.update_params(features, value_estimate)
        if self.vicarious_learning:
            self.update_params(term_features, term_reward)

    def perform_montecarlo_updates(self):
        if self.monte_carlo_updates:
            for i in range(len(self.update_features) - 1):
                self.update_params(
                    self.update_features[i], np.sum(self.update_rewards[i:])
                )

    def perform_end_episode_updates(self, env, features, reward, taken_path):
        selected_action = 0
        delay = env.get_feedback({"taken_path": taken_path, "action": selected_action})
        pr = self.get_pseudo_reward(env)
        value_estimate = reward + pr - self.delay_scale * delay
        self.update_params(features, value_estimate)
        self.update_rewards.append(
            value_estimate
        )  # This line has been changed. This only affects montecarlo models
        self.perform_montecarlo_updates()

    def learn_from_path(self, env, path):
        if self.path_learn:
            for node in path:
                f = env.get_action_state(node)
                self.update_params(f, env.present_trial.node_map[node].value)
                env.step(node)

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
            selected_action_prob = mp.quad(
                lambda x: norm_integrate(x, action_index, means, sigmas),
                [lb, ub],
                maxdegree=10,
            )
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
            self.store_action_likelihood(env, action)
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
            else:
                self.update_features.append(features)
                if self.learn_from_path_boolean:
                    self.learn_from_path(env, taken_path)
                self.perform_end_episode_updates(env, features, reward, taken_path)
            return action, reward, done, taken_path
        else:  # Should this model learn from the termination action?
            reward = 0
            taken_path = None
            if self.compute_likelihood:
                reward, taken_path, done = trial_info["participant"].make_click()
            if self.learn_from_path_boolean:
                self.learn_from_path(env, trial_info["taken_path"])
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
