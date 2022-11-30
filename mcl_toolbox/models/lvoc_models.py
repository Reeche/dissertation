from collections import defaultdict
import random

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.utils.learning_utils import (break_ties_random,
                                              estimate_bayes_glm,
                                              get_log_norm_cdf,
                                              get_log_norm_pdf, norm_integrate,
                                              rows_mean, sample_coeffs)

import time
import copy

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
        self.last_mean = []
        self.last_precision = []
        self.trial_action_learn_features = []
        self.trial_costs = 0
        self.bounds = []
        self.precisions = []
        self.action_likelihood_times = []

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        self.mean = self.init_weights
        self.precision = np.diag([1 / self.standard_dev ** 2] * self.num_features)
        self.last_mean = np.zeros_like(self.mean)
        self.last_precision = np.zeros_like(self.precision)
        self.bounds = []
        self.precisions = []
        self.gamma_a = 1
        self.gamma_b = 1
        self.trial_action_learn_features = []
        self.trial_costs = 0
        self.action_log_probs = []
        self.action_likelihood_times = []

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
       # print("Performing param update")
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

        # if computing participant click likelihood, use participant actions
        # instead of forward-simulating actions from model params
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
        print("Performing action update")
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
        if reward is None:
            return
        print("Performing end episode updates: {}".format(reward))
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

    def compute_all_likelihood_and_store(self, env, given_action):
        print("\nGetting all action likelihood times")
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
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
        # Distributions from which the weights are drawn?
        means = dists[:, 0]
        sigmas = np.sqrt(dists[:, 1])

        # Very important to select good bounds for proper sampling.
        ub = np.max(means + 5 * sigmas)
        lb = np.min(means - 5 * sigmas)
        self.bounds.append(ub-lb)
        # print([lb,ub], ub-lb)
        for idx, action in enumerate(available_actions):

            if num_available_actions == 1:
                selected_action_prob = 1
            else:
                selected_action_prob = mp.quad(
                    lambda x: norm_integrate(x, idx, means, sigmas),
                    [lb, ub],
                    maxdegree=10,
                )
            # print("Computing selected action prob: {0:0.3f}".format(end - start))
            eps = self.eps

            log_prob = float(
                str(
                    mp.log(
                        (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
                    )
                )
            )

            if action == given_action:
                self.action_log_probs.append(log_prob)

    def get_all_action_likelihood_times(self, env, given_action, action_time, action_prob):
        print("\nGetting all action likelihood times")
        available_actions = env.get_available_actions()
        current_time_dict = {}
        if self.no_term:
            available_actions.remove(0)
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
        # Distributions from which the weights are drawn?
        means = dists[:, 0]
        sigmas = np.sqrt(dists[:, 1])

        # Very important to select good bounds for proper sampling.
        ub = np.max(means + 5 * sigmas)
        lb = np.min(means - 5 * sigmas)


        # print([lb,ub], ub-lb)
        for idx, action in enumerate(available_actions):
            # if action == given_action:
            #     current_time_dict[action] = {
            #         "time": action_time,
            #         "prob": float(action_prob)
            #     }
            #     print("Action: {0}, Prob: {1:0.5f}, Time: {2:0.3f} Selected".format(action, float(action_prob), action_time))
            #     continue
            start = time.time()
            if num_available_actions == 1:
                selected_action_prob = 1
            else:
                selected_action_prob = mp.quad(
                    lambda x: norm_integrate(x, idx, means, sigmas),
                    [lb, ub],
                    maxdegree=10,
                )
            # print("Computing selected action prob: {0:0.3f}".format(end - start))
            eps = self.eps

            log_prob = float(
                str(
                    mp.log(
                        (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
                    )
                )
            )
            end = time.time()
            if(action == given_action):
                # print("Selected action")
                # print("Num available actions: {0}, Action index: {1}".format(num_available_actions,idx))
                print("Action: {0}, Prob: {1:0.5f}, Time: {2:0.3f} {3:0.3f} {4}"
                      .format(action,
                              float(selected_action_prob),
                              end-start,
                              action_time,
                              float(selected_action_prob)==action_prob))
                #print("Time difference: {0:0.3f}".format(action_time - (end-start)))
                #print("Probs same: {}".format(selected_action_prob == action_prob))
            else:
                print("Action: {0}, Prob: {1:0.5f}, Time: {2:0.3f}".format(action, float(selected_action_prob), end-start))
            current_time_dict[action] = {
                "time": end - start,
                "prob": float(selected_action_prob)
            }
        current_time_dict["selected"] = given_action
        self.action_likelihood_times.append(current_time_dict)

    def store_action_likelihood(self, env, given_action):
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
        action_index = available_actions.index(given_action)
        num_available_actions = len(available_actions)
        if((self.mean == self.last_mean).all()):
            print("Means Same")
            pass
        else:
            print("Means Different")
            self.last_mean = self.mean
        if((self.precision == self.last_precision).all()):
            print("Precisions Same")
            pass
        else:
            print("Precisions Different")
            self.last_precision = self.precision

        feature_vals = self.get_action_features(env)
        dists = np.zeros((num_available_actions, 2))

        # Should be diagonal if variance of posterior is diagonal
        cov = np.linalg.inv(self.precision*self.num_samples)

        self.precisions.append(np.sum(self.precision))
        for index, action in enumerate(available_actions):
            computed_features = feature_vals[action]

            # E[Q_hat] for action in belief state
            dists[index][0] = np.dot(computed_features, self.mean)

            # Var[Q_hat] for action in belief state
            dists[index][1] = np.dot(
                np.dot(computed_features, cov), computed_features.T
            )

        # What is driving Var[Q_hat] to systematically increase?
        # Hypotheses:
        #   Less frequent updates => fewer strategy changes => habitual feature that counts how often a click
        #       is made goes up, driving up the product of the features

        # Sanity check - change in uncertainty about the values of the feature weights - should go down rather than up
        #       (self.precision should increase)
        #   If there is increase in precision (normal):
        #       Possible that increase in sigma comes from changing features
        #       Plot values of features - does it make sense that this feature is going up?
        #           If some feature values are going up even if they shouldn't:
        #               Features might be implemented wrongly
        #       Maybe it's acting more habitually
        #   If there is decrease in precision (abnormal):
        #       Something is wrong
        #       Take a look at whether precision updating is correctly implemented

        # self.precision = 1/sigma**2, where sigma is the standard deviation of the posterior on the weights
        # self.mean = Mean vector of Gaussian posterior distribution over the weights
        # Distributions from which the weights are drawn?
        #
        # means = E[Q_hat]
        # sigmas = sqrt(Var[Q_hat])
        means = dists[:, 0]
        sigmas = np.sqrt(dists[:, 1])

        # Very important to select good bounds for proper sampling.
        ub = np.max(means + 5 * sigmas)
        lb = np.min(means - 5 * sigmas)

        self.bounds.append(ub-lb)

        start = time.time()
        if num_available_actions == 1:
            selected_action_prob = 1
        else:

            # Set limits to fine-grainedness
            selected_action_prob = mp.quad(
                lambda x: norm_integrate(x, action_index, means, sigmas),
                [lb, ub],
                maxdegree=10,
            )
        end = time.time()
        # print("Computing selected action prob: {0:0.3f}".format(end - start))
        eps = self.eps

        log_prob = float(
            str(
                mp.log(
                    (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
                )
            )
        )
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index], selected_action_prob

    def take_action(self, env, trial_info):
        action, features = self.get_action_details(env, trial_info)
        delay = env.get_feedback({"action": action})
        prob = 0
        if self.compute_likelihood:
            pi = trial_info["participant"]
            start = time.time()
            if self.compute_all_likelihoods_boolean:
                self.compute_all_likelihood_and_store(env, given_action=action)
            else:
                _, _, prob = self.store_action_likelihood(env, action)
            end = time.time()
            #self.get_all_action_likelihood_times(env, action, end-start, prob)
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
            start = time.time()

            (
                s_next,
                action,
                reward,
                done,
                taken_path,
                delay,
                features,
            ) = self.take_action(env, trial_info)
            print("Reward: {}".format(reward))
            if not self.compute_likelihood:
                print("Stepping  through env with action: {}, reward: {}".format(action, reward))
            end = time.time()
            print("Taking action: {0:0.3f}s".format(end - start))
            if not done:
                a_next, next_features = self.get_action_details(env, trial_info)
                # 2 - Save action features to learn from them only if terminal reward present
                #   Learn from all actions individually at once at the end of the episode
                if self.learn_from_actions == 2:
                    learn_from_this_action = False
                    self.trial_action_learn_features.append(
                        (
                            copy.deepcopy(features),
                            copy.deepcopy(env),
                            copy.deepcopy(next_features),
                            reward - self.delay_scale * delay,
                            copy.deepcopy(term_features),
                            copy.deepcopy(term_reward),
                        )
                    )
                else:
                    # 0 - don't learn from actions individually
                    # 1 - learn from all actions individually after they are taken
                    learn_from_this_action = random.random() < self.learn_from_actions
                self.trial_costs += reward
                if learn_from_this_action:
                    start = time.time()
                    self.update_features.append(features)
                    self.perform_action_updates(
                        env,
                        next_features,
                        reward - self.delay_scale * delay,
                        term_features,
                        term_reward,
                        features,
                    )
                    end = time.time()
                    # print("Learning from actions: {0:0.3f}s".format(end - start))
            elif reward is not None:
                start = time.time()

                # 2 - Learn from all individual actions at the end
                #   Terminal reward does not include the click costs
                if self.learn_from_actions == 2:
                    reward_to_learn = reward
                    for (
                            features,
                            new_env,
                            next_features,
                            reward,
                            term_features,
                            term_reward
                    ) in self.trial_action_learn_features:
                        self.update_features.append(features)
                        self.perform_action_updates(
                            new_env,
                            next_features,
                            reward - self.delay_scale * delay,
                            term_features,
                            term_reward,
                            features,
                        )
                else:
                    # 0 - don't learn from individual actions, terminal reward includes click costs
                    # 1 - learn from all actions regardless of presence of terminal reward,
                    #       terminal reward does not include click costs
                    reward_to_learn = reward + (1-self.learn_from_actions) * self.trial_costs
                self.update_features.append(features)
                if self.learn_from_path_boolean:
                    self.learn_from_path(env, taken_path)
                self.perform_end_episode_updates(env, features, reward_to_learn, taken_path)
                end = time.time()
                # print("performing end ep updates: {0:0.3f}s".format(end - start))
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
        # print("Running simulation")
        self.init_model_params()
        self.compute_likelihood = compute_likelihood
        env.reset()
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        if compute_likelihood:
            get_log_norm_pdf.cache_clear()
            get_log_norm_cdf.cache_clear()
        for trial_num in range(num_trials):
            print(trial_num)
            self.previous_best_paths = []
            self.num_actions = len(env.get_available_actions())
            trials_data["w"].append(self.get_current_weights())
            self.update_rewards, self.update_features = [], []
            self.trial_costs = 0
            self.trial_action_learn_features = []
            actions, rewards, self.term_rewards = [], [], []
            done = False

            while not done:
                action, reward, done, taken_path = self.act_and_learn(
                    env, trial_info={"participant": participant}
                )
                rewards.append(0 if reward is None else reward)
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
