from collections import defaultdict

import numpy as np

from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.models.rssl_models import RSSL
from mcl_toolbox.utils.sequence_utils import get_clicks


class SDSS(Learner):
    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self._bandit_params = np.array(params["bandit_params"])
        self._num_strategies = int(self._bandit_params.shape[0] / 2)
        self._threshold = params["bernoulli_threshold"]
        self._learner = attributes["learner"]

        self.learners = []
        for i in range(self._num_strategies):
            self.learners.append(self._learner(params, attributes))

        # In SDSS, by design RSSL doesn't learn using PRs or feedback
        self.rssl = RSSL({"priors": self._bandit_params, "pr_weight": 1}, attributes)

        if "strategy_weights" in attributes:
            self._strategy_weights = np.array(attributes["strategy_weights"])
        else:
            # Assuming blank slate
            self._strategy_weights = np.zeros(
                (self._num_strategies, self.num_features)
            )

        self.action_log_probs = []

        self.init_ts_params()

    def init_ts_params(self):
        for i, learner in enumerate(self.learners):
            learner.init_weights = self._strategy_weights[i]
            learner.init_model_params()

    def update_bernoulli_params(self, reward, strategy_index):
        num_strategies = self._num_strategies
        normalized_reward = (reward - self.rssl.lower_limit) / (
            self.rssl.upper_limit - self.rssl.lower_limit
        )
        params = self.rssl.priors
        alpha = params[strategy_index]
        beta = params[strategy_index + num_strategies]
        if not self.rssl.stochastic_updating:
            self.rssl.priors[strategy_index] += normalized_reward
            self.rssl.priors[strategy_index + num_strategies] += 1 - normalized_reward
        else:
            choice = np.random.binomial(n=1, p=normalized_reward) == 1
            if choice:
                self.rssl.priors[strategy_index] += 1
            else:
                self.rssl.priors[strategy_index + num_strategies] += 1
        C = self._threshold
        if alpha + beta >= C:
            self.rssl.priors[strategy_index] *= C / (C + 1)
            self.rssl.priors[strategy_index + num_strategies] *= C / (C + 1)

    def get_learner_details(self, env, strategy_num):
        """Select the best action and store the action features"""
        env.reset_trial()
        learner = self.learners[strategy_num]
        learner.num_actions = len(env.get_available_actions())
        learner.update_features = []
        learner.update_rewards = []
        learner.term_rewards = []
        learner.previous_best_paths = []
        rewards = []
        actions = []
        done = False
        while not done:
            action, reward, done, taken_path = learner.act_and_learn(env)
            actions.append(action)
            rewards.append(reward)
        return actions, rewards

    def apply_strategy(self, env, strategy_num):
        strategy_weights = self.learners[strategy_num].get_weights()
        env.reset_trial()
        trial = env.present_trial
        env.reset_trial()
        actions = get_clicks(
            trial, self.features, strategy_weights, self.normalized_features
        )
        f_list = []
        r_list = []
        env.reset_trial()
        for action in actions:
            f = env.get_feature_state(self.features, self.normalized_features)[action]
            _, r, _, _ = env.step(action)
            r_list.append(r)
            f_list.append(f)
        return actions, f_list, r_list

    def simulate(self, env, compute_likelihood=False, participant=None):
        env.reset()
        self.init_ts_params()
        self.action_log_probs = []
        num_trials = env.num_trials
        trials_data = defaultdict(list)
        for trial_num in range(num_trials):
            chosen_strategy = self.rssl.select_strategy()
            actions, r_list = self.get_learner_details(env, chosen_strategy)
            reward = np.sum(r_list)
            trials_data["costs"].append(r_list)
            self.update_bernoulli_params(reward, chosen_strategy)
            trials_data["r"].append(reward)
            trials_data["w"].append(self._strategy_weights[chosen_strategy])
            trials_data["a"].append(actions)
            trials_data["s"].append(chosen_strategy)
            env.get_next_trial()
        if self.action_log_probs:
            trials_data["loss"] = -np.sum(self.action_log_probs)
        else:
            trials_data["loss"] = None
        return dict(trials_data)
