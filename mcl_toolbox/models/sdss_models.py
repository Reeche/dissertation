from collections import defaultdict

import numpy as np
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.models.rssl_models import RSSL
from mcl_toolbox.utils.learning_utils import get_normalized_feature_values, get_log_beta_pdf, \
    get_log_beta_cdf
from mcl_toolbox.utils.sequence_utils import get_clicks


# TODO:
# Implement Gaussian DS - Requires knowledge of how to remove the influence of past observations
# How to have priors for strategies at the selector level?

class SDSS(Learner):
    # TODO:
    # Implement delay scale if required
    # Change it so that it takes all versions of the LVOC
    def __init__(self, params, attributes):

        self._bandit_params = np.array(params['bandit_params'])
        self._num_strategies = int(self._bandit_params.shape[0]/2)
        self._threshold = params['bernoulli_threshold']
        self._features = attributes['features']
        self._num_features = len(self._features)
        self._use_pr = attributes['use_pseudo_rewards']
        self._learner = attributes['learner']

        self.learners = []
        for i in range(self._num_strategies):
            self.learners.append(self._learner(params, attributes))
        
        strategy_space = range(1, self._num_strategies + 1)
        self.rssl = RSSL({'priors': self._bandit_params}, attributes)
                                    
        if 'strategy_weights' in params:
            self._strategy_weights = np.array(params['strategy_weights'])
        else:
            self._strategy_weights = np.random.rand(self._num_strategies, self._num_features)

        self.init_ts_params()

    def init_ts_params(self):
        for i, learner in enumerate(self.learners):
            learner.init_weights = self._strategy_weights[i]
            learner.init_model_params()

    def update_bernoulli_params(self, reward, strategy_index):
        num_strategies = self._num_strategies
        normalized_reward = (reward - self.rssl.lower_limit) / \
                (self.rssl.upper_limit-self.rssl.lower_limit)
        params = self.rssl.priors
        alpha = params[strategy_index]
        beta = params[strategy_index + num_strategies]
        if not self.rssl.stochastic_updating:
            self.rssl.priors[strategy_index] += normalized_reward
            self.rssl.priors[strategy_index + num_strategies] += (1-normalized_reward)
        else:
            choice = (np.random.binomial(n=1, p=normalized_reward) == 1)
            if choice:
                self.rssl.priors[strategy_index] += 1
            else:
                self.rssl.priors[strategy_index+ num_strategies] += 1
        C = self._threshold
        if alpha + beta >= C:
            self.rssl.priors[strategy_index] *= (C/(C+1))
            self.rssl.priors[strategy_index + num_strategies] *= (C/(C+1))

    def get_learner_details(self, env, strategy_num):
        """Select the best action and store the action features"""
        env.reset_trial()
        learner = self.learners[strategy_num]
        learner.num_actions = len(env.get_available_actions())
        learner.update_features = []
        learner.update_rewards = []
        rewards = []
        actions = []
        while(1):
            action, reward, done, taken_path = learner.act_and_learn(env)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        return actions, rewards
    
    def apply_strategy(self, env, strategy_num):
        strategy_weights = self.learners[strategy_num].get_weights()
        env.reset_trial()
        trial = env.present_trial
        env.reset_trial()
        actions = get_clicks(trial, self._features, strategy_weights, self._normalized_features)
        f_list = []
        r_list = []
        env.reset_trial()
        for action in actions:
            f = get_normalized_feature_values(trial.node_map[action].compute_termination_feature_values(self._features), self._features, self._normalized_features)
            _, r, _, _ = env.step(action)
            r_list.append(r)
            f_list.append(f)
        return actions, f_list, r_list
    
    def simulate(self, env, compute_likelihood=False, participant=None):
        env.reset()
        get_log_beta_cdf.cache_clear()
        get_log_beta_pdf.cache_clear()
        self.init_ts_params()
        self.action_log_probs = []
        num_trials = env.num_trials
        trials_data = defaultdict(list)
        for trial_num in range(num_trials):
            self._num_actions = len(env.get_available_actions())
            chosen_strategy = self.rssl.select_strategy()
            actions, r_list = self.get_learner_details(env, chosen_strategy)
            reward = np.sum(r_list)
            self.update_bernoulli_params(reward, chosen_strategy)
            trials_data['r'].append(reward)
            trials_data['w'].append(self._strategy_weights[chosen_strategy])
            trials_data['a'].append(actions)
            trials_data['s'].append(chosen_strategy)
            env.get_next_trial()
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)