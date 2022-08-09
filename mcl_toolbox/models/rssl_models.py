from collections import defaultdict

import mpmath as mp
import numpy as np

from models.base_learner import Learner #for running on the server, remove mcl_toolbox. part
from utils.learning_utils import ( #for running on the server, remove mcl_toolbox. part
    beta_integrate,
    get_log_beta_cdf,
    get_log_beta_pdf,
    get_log_norm_cdf,
    get_log_norm_pdf,
    norm_integrate,
)
from utils.planning_strategies import strategy_dict #for running on the server, remove mcl_toolbox. part

precision_epsilon = 1e-4
quadrature_max_degree = 1e5


def clear_cache():
    get_log_beta_cdf.cache_clear()
    get_log_beta_pdf.cache_clear()
    get_log_norm_cdf.cache_clear()
    get_log_norm_pdf.cache_clear()


class RSSL(Learner):
    """Base class of the RSSL models with different priors"""

    # TODO:
    # Give ability to change source of strategies
    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.priors = params["priors"]
        self.strategy_space = attributes["strategy_space"]
        self.num_strategies = len(self.strategy_space)
        self.upper_limit = 500  # Buffer for subjective cost
        self.lower_limit = -500  # Buffer for subjective cost
        self.gaussian = attributes["is_gaussian"]
        if self.gaussian:
            self.priors = np.exp(self.priors)
        self.variance = 1
        if "gaussian_var" in attributes:
            self.variance = np.exp(attributes["gaussian_var"])
        self.stochastic_updating = attributes["stochastic_updating"]
        self.action_log_probs = False
        if "strategy_probs" in attributes:
            self.strategy_probs = attributes["strategy_probs"]

    def gaussian_likelihood(
        self, strategy_index
    ):  # Numerical integration to compute the likelihood under the gaussian distribution
        priors = self.priors
        num_strategies = self.num_strategies
        means = priors[:num_strategies]
        sigmas = np.sqrt(priors[num_strategies:])
        max_val = np.max(means + 5 * sigmas)
        min_val = np.min(means - 5 * sigmas)
        likelihood = mp.quad(
            lambda x: norm_integrate(x, strategy_index, means, sigmas),
            [min_val, max_val],
        )
        return likelihood

    def bernoulli_likelihood(
        self, strategy_index
    ):  # Numerical integration to compute the likelihood under the beta distribution
        priors = self.priors
        num_strategies = self.num_strategies
        alphas = priors[:num_strategies]
        betas = priors[num_strategies:]
        max_val = 1
        min_val = 0
        likelihood = mp.quad(
            lambda x: beta_integrate(x, strategy_index, alphas, betas),
            [min_val, max_val],
        )
        return likelihood

    def get_max_likelihoods(self, strategy_index):
        if self.gaussian:
            return self.gaussian_likelihood(strategy_index)
        else:
            return self.bernoulli_likelihood(strategy_index)

    def bernoulli_choice(self):
        priors = self.priors
        values = np.zeros(self.num_strategies)
        for strategy_num in range(self.num_strategies):
            values[strategy_num] = np.random.beta(
                priors[strategy_num] + 1, priors[strategy_num + self.num_strategies] + 1
            )
        return np.argmax(values)

    def gaussian_choice(self):
        priors = self.priors
        num_strategies = self.num_strategies
        values = np.zeros(num_strategies)
        for strategy_num in range(num_strategies):
            values[strategy_num] = np.random.normal(
                priors[strategy_num], np.sqrt(priors[strategy_num + num_strategies])
            )
        return np.argmax(values)

    def update_bernoulli_params(self, reward, strategy_index):
        normalized_prob = (reward - self.lower_limit) / (
            self.upper_limit - self.lower_limit
        )
        priors = self.priors
        if self.stochastic_updating:
            choice = np.random.binomial(n=1, p=normalized_prob) == 1
            if choice:
                priors[strategy_index] += 1
            else:
                priors[strategy_index + self.num_strategies] += 1
        else:
            priors[strategy_index] += normalized_prob
            priors[strategy_index + self.num_strategies] += 1 - normalized_prob

    def update_gaussian_params(self, reward, strategy_index):
        var = self.variance
        num_strategies = self.num_strategies
        priors = self.priors
        priors[strategy_index] = (
            priors[strategy_index + num_strategies] * reward
            + priors[strategy_index] * var
        ) / (priors[strategy_index + num_strategies] + var)
        priors[strategy_index + num_strategies] = (
            priors[strategy_index + num_strategies]
            * var
            / (priors[strategy_index + num_strategies] + var)
        )

    def update_params(self, reward, strategy_index):
        if self.is_null:
            return
        if self.gaussian:
            self.update_gaussian_params(reward, strategy_index)
        else:
            self.update_bernoulli_params(reward, strategy_index)

    def select_strategy(self):
        if self.gaussian:
            strategy_index = self.gaussian_choice()
        else:
            strategy_index = self.bernoulli_choice()
        return strategy_index

    def apply_strategy(self, env, trial, strategy_index, info=None):
        strategy_space = self.strategy_space[strategy_index]
        taken_path = None
        if info is not None:
            if "actions" in info:
                actions = info["actions"]
        else:
            actions = strategy_dict[strategy_space](trial)
        env.reset_trial()
        r_list = []
        delays = []
        prs = []
        for action in actions:
            delay = env.get_feedback({"action": action})
            self.store_best_paths(env)
            _, r, _, taken_path = env.step(action)
            r_list.append(r)
            delays.append(self.delay_scale * delay)
            prs.append(self.get_pseudo_reward(env))
        if info is not None:
            taken_path = info["taken_path"]
            r_list = info["rewards"]
        delay = env.get_feedback({"action": 0, "taken_path": taken_path})
        delays.append(delay)
        info = {"taken_path": taken_path, "delays": delays, "prs": prs}
        return actions, r_list, info

    def compute_log_likelihood(self, chosen_strategy):
        strategy_index = self.strategy_space.index(chosen_strategy)
        strategy_likelihood = self.get_max_likelihoods(strategy_index)
        return mp.log(strategy_likelihood)

    def simulate(self, env, compute_likelihood=False, participant=None):
        env.reset()
        clear_cache()
        if compute_likelihood:
            self.action_log_probs = True
        action_log_probs = []
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        for trial_num in range(num_trials):
            trial = env.trial_sequence.trial_sequence[trial_num]
            self.previous_best_paths = []
            info = None
            if compute_likelihood:
                clicks = participant.clicks[trial_num]
                rewards = participant.rewards[trial_num]
                chosen_strategy = participant.strategies[trial_num]
                strategy_index = self.strategy_space.index(chosen_strategy)
                ll = self.strategy_probs[trial_num]
                log_prob = self.compute_log_likelihood(chosen_strategy)
                action_log_probs.append(ll + float(log_prob))
                info = {
                    "taken_path": participant.get_trial_path(),
                    "actions": clicks,
                    "rewards": rewards,
                }
                participant.current_trial += 1
            else:
                strategy_index = self.select_strategy()
            clicks, r_list, info = self.apply_strategy(
                env, trial, strategy_index, info=info
            )
            reward = np.sum(r_list)
            trials_data["costs"].append(r_list)
            trials_data["taken_paths"].append(info["taken_path"])
            update_reward = reward.copy()
            update_reward -= (len(r_list) - 1) * self.subjective_cost
            update_reward -= np.sum(info["delays"])
            update_reward += np.sum(info["prs"])
            self.update_params(update_reward, strategy_index)
            trials_data["r"].append(reward)
            chosen_strategy = self.strategy_space[strategy_index]
            trials_data["s"].append(chosen_strategy)
            trials_data["a"].append(clicks)
            env.get_next_trial()
        # add trial ground truths
        trials_data["envs"] = env.ground_truth
        if self.action_log_probs:
            trials_data["loss"] = -np.sum(action_log_probs)
        else:
            trials_data["loss"] = None
        return dict(trials_data)


class BernoulliRSSL(RSSL):
    """RSSL model with bernoulli priors"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)


class GaussianRSSL(RSSL):
    """ RSSL model with Gaussian priors"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.gaussian = True


class NullBernoulliRSSL(BernoulliRSSL):
    """ Bernoulli RSSL without learning """

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.is_null = True


class NullGaussianRSSL(GaussianRSSL):
    """ Gaussian RSSL without learning """

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.is_null = True
        self.gaussian = True
