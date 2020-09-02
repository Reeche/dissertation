import numpy as np
import mpmath as mp
from base_learner import Learner
from collections import defaultdict
from learning_utils import sample_coeffs, rows_mean, estimate_bayes_glm, \
    get_normalized_feature_values, break_ties_random, \
    get_log_norm_pdf, get_log_norm_cdf
from rl_models import integrate

class LVOC(Learner):
    """Base class of the LVOC model"""

    def __init__(self, params, attributes):
        super().__init__()
        self.standard_dev = np.exp(params['standard_dev'])
        self.num_samples = int(params['num_samples'])
        self.features = attributes['features']
        self.num_features = len(self.features)
        self.init_weights = params['priors']
        self.normalized_features = attributes['normalized_features']
        self.use_pseudo_rewards = attributes['use_pseudo_rewards']
        self.pr_weight = params['pr_weight']
        self.eps = max(0, min(params['eps'], 1))
        if 'delay_scale' in params:
            self.delay_scale = np.exp(params['delay_scale'])
        else:
            self.delay_scale = 0
        if 'subjective_cost' in params:
            self.subjective_cost = params['subjective_cost']
        else:
            self.subjective_cost = 0
        self.no_term = attributes['no_term']
        self.is_null = attributes['is_null']
        self.vicarious_learning = attributes['vicarious_learning']
        self.termination_value_known = attributes['termination_value_known']
        self.monte_carlo_updates = attributes['montecarlo_updates']
        self.init_model_params()
        self.term_rewards = []

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        self.mean = self.init_weights
        self.precision = np.diag([1 / (self.standard_dev) ** 2] * self.num_features)
        self.gamma_a = 1
        self.gamma_b = 1
        self.action_log_probs = []

    def get_current_weights(self):
        return self.mean.tolist()

    def get_best_paths(self, env):
        trial = env.present_trial
        expected_path_values = trial.get_path_expected_values()
        node_paths = trial.reverse_branch_map[0]
        best_paths = [k for k, v in expected_path_values.items(
        ) if v == max(expected_path_values.values())]
        return set(best_paths)

    def sample_weights(self):
        """Sample weights from the posterior distribution"""
        sampled_weights = sample_coeffs(
            self.mean, self.precision, self.gamma_a, self.gamma_b, self.num_samples)
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
            f, r, self.mean, self.precision, self.gamma_a, self.gamma_b)

    def get_action_features(self, env):
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
        num_available_actions = len(available_actions)
        feature_vals = np.zeros((self.num_actions, self.num_features))
        for index, action in enumerate(available_actions):
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self.features, adaptive_satisficing={})
            computed_features = get_normalized_feature_values(
                computed_features, self.features, self.normalized_features)
            feature_vals[action] = computed_features
        return np.array(feature_vals)

    def get_action_details(self, env):
        """Get the best action and its features in the given state"""
        current_trial = env.present_trial
        feature_vals = self.get_action_features(env)
        available_actions = env.get_available_actions()
        num_available_actions = len(available_actions)

        q = np.zeros(num_available_actions)
        sampled_weights = self.sample_weights()
        for index, action in enumerate(available_actions):
            q[index] = np.dot(sampled_weights, feature_vals[action])

        if self.termination_value_known:
            term_reward = current_trial.node_map[0].calculate_max_expected_return()
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

        if self.eps > 1:
            print(self.eps)
        random_action = np.random.binomial(1, p=self.eps)
        if random_action:
            best_action = np.random.choice(available_actions)
        return best_action, feature_vals[best_action]

    def get_action(self, env):
        """Select the best action and store the action features"""
        action, _ = self.get_action_details(env)
        return action

    # Verify this
    def perform_first_action_updates(self, env, reward, term_features, term_reward):
        self.update_rewards.append(reward - self.subjective_cost)
        if self.vicarious_learning:
            self.update_params(term_features, term_reward)

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
        best_paths = [branch_map[k][1:] for k, v in path_sums.items() if v == max_path_sum]
        self.previous_best_paths = best_paths

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
            return np.max(path_values)  # Changed from np.mean

    def perform_action_updates(self, env, next_features, reward, term_features, term_reward, features):
        q = np.dot(self.mean, next_features)
        pr = 0
        if self.use_pseudo_rewards:
            # comp_value = self.pr_weight*self.term_rewards[-1]
            comp_value = self.get_best_paths_expectation(env)
            mer = self.get_term_reward(env)
            pr = self.pr_weight * (mer - comp_value)
        self.update_rewards.append(reward + pr - self.subjective_cost)
        self.rpe = reward - q
        self.pseudo_reward = pr
        value_estimate = (q + (reward - self.subjective_cost) + pr)
        self.update_params(features, value_estimate)
        if self.vicarious_learning:
            self.update_params(term_features, term_reward)

    def perform_montecarlo_updates(self):
        if self.monte_carlo_updates:
            for i in range(len(self.update_features) - 1):
                self.update_params(
                    self.update_features[i], np.sum(self.update_rewards[i:]))

    def perform_end_episode_updates(self, env, features, reward, taken_path):
        delay = env.present_trial.get_action_feedback(taken_path)
        pr = 0
        if self.use_pseudo_rewards:
            # comp_value = self.pr_weight*self.term_rewards[-1]
            comp_value = self.get_best_paths_expectation(env)
            mer = self.get_term_reward(env)
            pr = self.pr_weight * (mer - comp_value)
        value_estimate = reward + pr - self.delay_scale * delay
        self.update_params(features, value_estimate)
        self.update_rewards.append(reward - self.delay_scale * delay)
        self.perform_montecarlo_updates()

    def act_and_learn(self, env, end_episode=False):
        if not end_episode:
            action, features = self.get_action_details(env)
            term_reward = self.get_term_reward(env)
            term_features = self.get_term_features(env)
            self.term_rewards.append(term_reward)
            self.store_best_paths(env)
            s_next, reward, done, info = env.step(action)
            taken_path = info
            taken_action = action

            if not done:
                a_next, next_features = self.get_action_details(env)
                self.update_features.append(features)
                self.perform_action_updates(
                    env, next_features, reward, term_features, term_reward, features)
            else:
                self.update_features.append(features)
                self.perform_end_episode_updates(env, features, reward, taken_path)
            return taken_action, reward, done, taken_path
        else:
            return None, None, None, None

    def take_action_and_learn(self, env, given_action, reward, next_action, trial_path):
        action_features = self.get_action_features(env)
        features = action_features[given_action]
        term_reward = self.get_term_reward(env)
        term_features = self.get_term_features(env)
        self.term_rewards.append(term_reward)
        self.store_best_paths(env)
        _, _, done, _ = env.step(given_action)
        if not done:
            features_next = self.get_action_features(env)[next_action]
            self.update_features.append(features)
            self.perform_action_updates(
                env, features_next, reward, term_features, term_reward, features)
        else:
            self.update_features.append(features)
            self.perform_end_episode_updates(env, features, reward, trial_path)

    def store_action_likelihood(self, env, given_action):
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
        num_available_actions = len(available_actions)
        feature_vals = self.get_action_features(env)

        dists = np.zeros((num_available_actions, 2))
        cov = np.linalg.inv(self.precision)
        for index, action in enumerate(available_actions):
            computed_features = feature_vals[action]
            dists[index][0] = np.dot(computed_features, self.mean)
            dists[index][1] = np.dot(np.dot(computed_features, cov), computed_features.T)

        means = dists[:, 0]
        sigmas = np.sqrt(dists[:, 1])

        # Very important to select good bounds for proper sampling.
        ub = np.max(means + 5 * sigmas)
        lb = np.min(means - 5 * sigmas)

        if num_available_actions == 1:
            probs = [1.0]
        else:
            probs = np.array(
                [mp.quad(lambda x: integrate(x, i, means, sigmas), [lb, ub]) for i in range(num_available_actions)])

        action_index = available_actions.index(given_action)
        selected_action_prob = probs[action_index]
        eps = self.eps
        log_prob = float(str(mp.log((1 - eps) * selected_action_prob + eps * (1 / num_available_actions))))
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]

    def simulate(self, env, compute_likelihood=False, participant=None):
        '''
        :param env: OpenAI-gym-compatible mouse lab environment using the GenericMouselabEnv class or a derived class
        :param compute_likelihood: whether or not to compute the likelihood for inputted participant data
        :param participant: #TODOCUMENT participant data description, dictionary that contains all_trials_data which contains keys actions, rewards and taken paths for each trial
        :return: a dictionary of trial data with keys w (weights), r (rewards), a (actions), info (list: [info #TODOCUMENT what is this, node value, pseudorward, rpe]), loss (#TODOCUMENT) which all contain an entry for each trial
        '''
        # TODO:
        # Fix update features
        # OPTIMIZE we can just remove the computer_likelihood and check if participant is none since that data is only used when that's True

        # Participant-level model priors
        self.init_model_params()

        # initialize trial data, get number of trials
        trials_data = defaultdict(list)
        num_trials = env.num_trials

        # reset env, clear pdf caches
        env.reset()
        get_log_norm_pdf.cache_clear()
        get_log_norm_cdf.cache_clear()

        for trial_num in range(num_trials):
            self.previous_best_paths = []
            self.num_actions = len(env.get_available_actions())
            trials_data['w'].append(self.get_current_weights())
            self.update_rewards, self.update_features = [], []
            actions, rewards, self.term_rewards = [], [], []
            if compute_likelihood:
                # extract participant data
                all_trials_data = participant.all_trials_data

                # step through participant data (using the method take_action_and_learn)
                trial_actions = all_trials_data['actions'][trial_num]
                trial_rewards = all_trials_data['rewards'][trial_num]
                trial_path = all_trials_data['taken_paths'][trial_num]
                for i in range(len(trial_actions)):
                    action = trial_actions[i]
                    reward = trial_rewards[i]
                    self.store_action_likelihood(env, action)
                    if i == len(trial_actions) - 1:
                        next_action = None
                    else:
                        next_action = trial_actions[i + 1]
                    actions.append(action)
                    rewards.append(reward)
                    self.take_action_and_learn(env, action, reward, next_action, trial_path)
            else:
                # step through using the method act_and_learn (#TODOCUMENT what is it doing - optimal action according to LVOC in get_action_details ?)
                done = False
                paths = []
                self.pseudo_reward = 0
                self.rpe = 0
                info_data = []
                while not done:
                    info = False
                    previous_best_path_value = self.get_term_reward(env)
                    previous_best_paths = self.get_best_paths(env)
                    action, reward, done, taken_path = self.act_and_learn(env)
                    current_best_path_value = self.get_term_reward(env)
                    current_best_paths = self.get_best_paths(env)
                    # info_value = current_best_path_value - previous_best_path_value
                    info_value = (previous_best_paths == current_best_paths)
                    node_value = env.present_trial.node_map[action].value
                    # print(f"Info: {info_value}, Node value: {node_value}, PR: {self.pseudo_reward}, RPE: {self.rpe}")
                    rewards.append(reward)
                    actions.append(action)
                    if not done:
                        _, f = self.get_action_details(env)
                        new_q = np.dot(self.mean, f)
                    else:
                        new_q = self.get_term_reward(env)
                        paths.append(taken_path)
                    if not done:
                        info_data.append(
                            [info_value, node_value, self.pseudo_reward, self.rpe + new_q])  # Here self.rpe = r-q
                trials_data['info'].append(info_data)
                trials_data['taken_paths'].append(paths)
            trials_data['r'].append(np.sum(rewards))
            trials_data['rewards'].append(rewards)
            trials_data['a'].append(actions)

            env.get_next_trial()

        trials_data["envs"] = env.ground_truth
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)


class simLVOC(LVOC):
    def __init__(self, params, attributes, max_iters=int(5e2)):
        super().__init__(params, attributes)
        self.max_iters = max_iters

    def store_action_likelihood(self, env, given_action):
        available_actions = env.get_available_actions()
        action_index = available_actions.index(given_action)
        num_available_actions = len(available_actions)
        if self.no_term:
            available_actions.remove(0)
        feature_vals = self.get_action_features(env)
        max_iters = self.max_iters
        count = 0
        for i in range(max_iters):
            a = self.get_action(env)
            if a == given_action:
                count += 1
        selected_action_prob = (count + 1) / (max_iters + num_available_actions)
        eps = self.eps
        log_prob = np.log((1 - eps) * selected_action_prob + eps * (1 / num_available_actions))
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]


class ibsLVOC(LVOC):
    def __init__(self, params, attributes, max_k_iters=int(5e4)):
        super().__init__(params, attributes)
        self.max_k_iters = max_k_iters

    def store_action_likelihood(self, env, given_action):
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        action_index = available_actions.index(given_action)
        num_available_actions = len(available_actions)
        if self.no_term:
            available_actions.remove(0)
        feature_vals = self.get_action_features(env)
        num_repeats = 1
        ll = 0
        max_k_iters = self.max_k_iters
        for _ in range(num_repeats):
            # IBS sampling
            k = 0
            while (1):
                k += 1
                s = self.get_action(env)
                if s == given_action or k == max_k_iters:
                    break
            # print(f"-----k is {k}------")
            if k != 1:
                L = np.array(list(range(1, k)))
                L = 1 / L
                ll += np.sum(L)
        selected_action_prob = np.exp(-ll / num_repeats)
        eps = self.eps
        log_prob = np.log((1 - eps) * selected_action_prob + eps * (1 / num_available_actions))
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]
