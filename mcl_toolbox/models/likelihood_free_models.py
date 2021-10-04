import numpy as np

from mcl_toolbox.models.lvoc_models import LVOC
from mcl_toolbox.models.rssl_models import RSSL


# Might not work as expected
class ibsLVOC(LVOC):
    def __init__(self, params, attributes, max_k_iters=int(5e4)):
        super().__init__(params, attributes)
        self.max_k_iters = max_k_iters

    def store_action_likelihood(self, env, given_action):
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
            while 1:
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
        log_prob = np.log(
            (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
        )
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]


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
        log_prob = np.log(
            (1 - eps) * selected_action_prob + eps * (1 / num_available_actions)
        )
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]


class IBSRSSL(RSSL):
    def __init__(self, params, attributes, max_k_iter=5e4):
        super().__init__(params, attributes)
        self.max_k_iter = max_k_iter

    def get_strategy_log_likelihood(self, strategy_index):
        num_repeats = 1
        ll = 0
        for _ in range(num_repeats):
            k = 0
            while 1:
                k += 1
                s = self.select_strategy()
                if s == strategy_index or k == self.max_k_iter:
                    break
            print(f"-----k is {k}------")
            if k != 1:
                L = np.array(list(range(1, k)))
                L = 1 / L
                ll += np.sum(L)
        return -ll / num_repeats  # LL

    def compute_log_likelihood(self, clicks, chosen_strategy):
        trial = None
        strategy_index = self.strategy_space.index(chosen_strategy)
        log_likelihood = self.get_strategy_log_likelihood(strategy_index)
        actions_strategy_log_likelihood = self.get_action_strategy_likelihood(
            trial, clicks, chosen_strategy, self.temperature
        )
        log_prob = log_likelihood + actions_strategy_log_likelihood
        return log_prob


class SimRSSL(RSSL):
    def __init__(
        self, priors, strategy_space, stochastic_updating=True, maxiter=int(1e3)
    ):
        super().__init__(priors, strategy_space, stochastic_updating)
        self.maxiter = maxiter

    def get_strategy_log_likelihood(self, strategy_index):
        count = np.zeros(self.num_strategies)
        for _ in range(self.max_iter):
            # Histogram sampling
            s = self.select_strategy()
            count[s] = count[s] + 1
        count += 1
        count /= count.sum()
        print(count, strategy_index, count[strategy_index])
        return np.log(count[strategy_index])

    def compute_log_likelihood(self, clicks, chosen_strategy):
        trial = None
        strategy_index = self.strategy_space.index(chosen_strategy)
        log_likelihood = self.get_strategy_log_likelihood(strategy_index)
        actions_strategy_log_likelihood = self.get_action_strategy_likelihood(
            trial, clicks, chosen_strategy, self.temperature
        )
        log_prob = log_likelihood + actions_strategy_log_likelihood
        return log_prob
