from collections import defaultdict

import mpmath as mp
import numpy as np
from base_learner import Learner
from learning_utils import get_normalized_features, pickle_load, \
    norm_integrate, beta_integrate, get_log_beta_pdf, get_log_beta_cdf, \
    get_log_norm_pdf, get_log_norm_cdf
from planning_strategies import strategy_dict

NS = 79
precision_epsilon = 1e-4
quadrature_max_degree = 1e5

class RSSL(Learner):
    """Base class of the RSSL models with different priors"""
    # The total likelihood under the RSSL model would be the likelihood
    # of the strategy being selected and that multiplied with the
    # probability of the click sequence under the selected strategy
    # Change the source of the strategies.
    # Add participant temperatures
    # Should we also fit the initial variance?
    def __init__(self, params, attributes):
        super().__init__()
        self.priors = params['priors']
        self.strategy_space = attributes['strategy_space']
        self.num_strategies = len(self.strategy_space)
        self.upper_limit = 200 # Buffer for subjective cost
        self.lower_limit = -200 # Buffer for subjective cost
        self.gaussian = attributes['is_gaussian']
        self.is_null = attributes['is_null']
        if self.gaussian:
            self.priors = np.exp(self.priors)
        # TODO:
        # Pass features and strategy weights to the model. For now, using a hack
        self.strategy_weights = pickle_load("data/microscope_weights.pkl")
        self.features = pickle_load("data/microscope_features.pkl")
        self.variance = 1 #Variance of the gaussian likelihood function
        self.stochastic_updating = attributes['stochastic_updating']
        if 'subjective_cost' in params:
            self.subjective_cost = params['subjective_cost']
        else:
            self.subjective_cost = 0

    def gaussian_max_likelihoods(self): # Numerical integration to compute the likelihood under the gaussian distribution
        priors = self.priors
        num_strategies = self.num_strategies
        means = priors[:num_strategies]
        sigmas = np.sqrt(priors[num_strategies:])
        max_val = np.max(means+5*sigmas)
        min_val = np.min(means-5*sigmas) 
        likelihoods = [mp.quad(lambda x: norm_integrate(x, i, means, sigmas),
                        [min_val, max_val]) for i in range(num_strategies)]
        return likelihoods
    
    def bernoulli_max_likelihoods(self): # Numerical integration to compute the likelihood under the beta distribution
        priors = self.priors
        num_strategies = self.num_strategies
        alphas = priors[:num_strategies]
        betas = priors[num_strategies:]
        max_val = 1
        min_val = 0
        likelihoods = [mp.quad(lambda x: beta_integrate(x, i, alphas, betas), [min_val, max_val]) for i in range(num_strategies)]
        return likelihoods
    
    def get_max_likelihoods(self):
        if self.gaussian:
            return self.gaussian_max_likelihoods()
        else:
            return self.bernoulli_max_likelihoods()

    def bernoulli_choice(self):
        priors = self.priors
        values = np.zeros(self.num_strategies)
        for strategy_num in range(self.num_strategies):
            values[strategy_num] = np.random.beta(
                priors[strategy_num] + 1, priors[strategy_num+self.num_strategies] + 1)
        return np.argmax(values)

    def gaussian_choice(self):
        priors = self.priors
        num_strategies = self.num_strategies
        values = np.zeros(num_strategies)
        for strategy_num in range(num_strategies):
            values[strategy_num] = np.random.normal(
                priors[strategy_num], np.sqrt(priors[strategy_num+num_strategies]))
        return np.argmax(values)

    def update_bernoulli_params(self, reward, strategy_index):
        num_strategies = self.num_strategies
        normalized_prob = (reward - self.lower_limit) / \
                (self.upper_limit-self.lower_limit)
        priors = self.priors
        if self.stochastic_updating:
            choice = (np.random.binomial(n=1, p=normalized_prob) == 1)
            if choice:
                priors[strategy_index] += 1
            else:
                priors[strategy_index+self.num_strategies] += 1
        else:
            priors[strategy_index] += normalized_prob
            priors[strategy_index+self.num_strategies] += 1 - normalized_prob

    def update_gaussian_params(self, reward, strategy_index):
        var = self.variance
        num_strategies = self.num_strategies
        priors = self.priors
        priors[strategy_index] = (priors[strategy_index+num_strategies]*reward +
                                           priors[strategy_index]*var)/(priors[strategy_index+num_strategies] + var)
        priors[strategy_index+num_strategies] = priors[strategy_index +
                                                        num_strategies]*var/(priors[strategy_index+num_strategies] + var)
    
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
    
    def apply_strategy(self, env, trial, strategy_index):
        S = self.strategy_space[strategy_index]
        actions = strategy_dict[S](trial)
        env.reset_trial()
        r_list = []
        for action in actions:
            _, r, _, _ = env.step(action)
            r_list.append(r)
        r_list = [r-self.subjective_cost for r in r_list[:-1]] + [r_list[-1]]
        return actions, r_list

    def get_action_strategy_likelihood(self, trial, actions, chosen_strategy, temperature):
        strategy_weights =  self.strategy_weights[chosen_strategy-1]*(1/temperature)
        #Only for now
        normalized_features = get_normalized_features("high_increasing")
        ll = compute_log_likelihood(trial, actions, self.features, strategy_weights, inv_t=False, normalized_features=normalized_features)
        return ll

    def compute_log_likelihood(self, clicks, chosen_strategy):
        likelihoods = self.get_max_likelihoods()
        strategy_index = self.strategy_space.index(chosen_strategy)
        strategy_likelihood = likelihoods[strategy_index]
        actions_strategy_log_likelihood = self.get_action_strategy_likelihood(trial, clicks, chosen_strategy, self.temperature)
        log_prob = float(str(actions_strategy_log_likelihood + mp.log(strategy_likelihood)))
        return log_prob

    def generate_trials_data(self, env, compute_likelihood, participant):
        action_log_probs = []
        all_trials_data = participant.all_trials_data
        first_trial_data = participant.first_trial_data
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        for trial_num in range(num_trials):
            trial = env.trial_sequence.trial_sequence[trial_num]
            if compute_likelihood:
                clicks = all_trials_data['actions'][trial_num]
                rewards = all_trials_data['rewards'][trial_num]
                chosen_strategy = all_trials_data['strategies'][trial_num]
                log_prob = self.compute_log_likelihood(clicks, chosen_strategy)
                action_log_probs.append(log_prob)
                reward = np.sum(rewards)
                self.update_params(reward, strategy_index)
            else:
                strategy_index = self.select_strategy()
                clicks, r_list = self.apply_strategy(env, trial, strategy_index)
                reward = np.sum(r_list)
                self.update_params(reward, strategy_index)
            trials_data['r'].append(reward)
            chosen_strategy = self.strategy_space[strategy_index]
            trials_data['s'].append(chosen_strategy)
            trials_data['w'].append(self.strategy_weights[chosen_strategy-1])
            trials_data['a'].append(clicks)
            env.get_next_trial()
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(action_log_probs)
        else:
            trials_data['loss'] = None
        return trials_data

    def simulate(self, env, compute_likelihood=False, participant=None):
        env.reset()
        self.action_log_probs = False
        if compute_likelihood:
            self.action_log_probs = True
        get_log_beta_cdf.cache_clear()
        get_log_beta_pdf.cache_clear()
        get_log_norm_cdf.cache_clear()
        get_log_norm_pdf.cache_clear()
        self.temperature = 1
        if hasattr(participant, 'temperature'):
            self.temperature = participant.temperature
        trials_data = self.generate_trials_data(env, compute_likelihood, participant)
        return dict(trials_data)

class IBSRSSL(RSSL):

    def __init__(self, params, attributes, max_k_iter=5e4):
        super().__init__(params, attributes)
        self.max_k_iter = max_k_iter

    def get_strategy_log_likelihood(self, strategy_index):
        params = self.params
        num_repeats = 1
        ll = 0
        for _ in range(num_repeats):
            k = 0
            while(1):
                k += 1
                s = self.select_strategy()
                if s == strategy_index or k == self.max_k_iter:
                    break
            print(f"-----k is {k}------")
            if k!=1:
                L = np.array(list(range(1, k)))
                L = 1/L
                ll += np.sum(L)
        return -ll/num_repeats #LL

    def compute_log_likelihood(self, clicks, chosen_strategy):
        strategy_index = self.strategy_space.index(chosen_strategy)
        log_likelihood = self.get_strategy_log_likelihood(strategy_index)
        actions_strategy_log_likelihood = self.get_action_strategy_likelihood(trial, clicks, chosen_strategy, self.temperature)
        log_prob = log_likelihood + actions_strategy_log_likelihood
        return log_prob

class SimRSSL(RSSL):

    def __init__(self, priors, strategy_space, stochastic_updating=True, maxiter=int(1e3)):
        super().__init__(priors, strategy_space, stochastic_updating)
        self.maxiter = maxiter

    def get_strategy_log_likelihood(self, strategy_index):
        params = self.params
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
        strategy_index = self.strategy_space.index(chosen_strategy)
        log_likelihood = self.get_strategy_log_likelihood(strategy_index)
        actions_strategy_log_likelihood = self.get_action_strategy_likelihood(trial, clicks, chosen_strategy, self.temperature)
        log_prob = log_likelihood + actions_strategy_log_likelihood
        return log_prob

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