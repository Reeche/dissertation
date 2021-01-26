import os
import gym
import torch
import numpy as np
import scipy as sp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mpmath as mp
import seaborn as sns
import hyperopt
import inspect
from torch import autograd
from torch.autograd import Variable
from learning_utils import *
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from collections import defaultdict
from hyperopt import hp, fmin, tpe, Trials
from functools import partial
from generic_mouselab import GenericMouselabEnv
from sequence_utils import compute_log_likelihood, get_clicks
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import logsumexp
from numba import jit
from functools import lru_cache
from planning_strategies import strategy_dict
from math import sqrt

normalize = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
             2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 72.0, 2.0, 2.0]
precision_epsilon = 1e-4
quadrature_max_degree = 1e5
sns.set_style('whitegrid')

NS = 79

#TODO:
# Implement Gaussian DS - Requires knowledge of how to remove the influence of past observations

@lru_cache(maxsize=None)
def get_log_pdf(y, m, v):
    return mp.log(mp.npdf(y, m, v))

@lru_cache(maxsize=None)
def get_log_cdf(y, m, v):
    return mp.log(mp.ncdf(y, m, v))

@lru_cache(maxsize=None)
def get_log_beta_pdf(x, a, b):
    log_x = mp.log(x)
    log_ox = mp.log(1-x)
    res = (a-1)*log_x + (b-1)*log_ox + mp.loggamma(a+b) - mp.loggamma(a) - mp.loggamma(b)
    return res

@lru_cache(maxsize=None)
def get_log_beta_cdf(x, a, b):
    res = mp.log(mp.betainc(a, b, x2=x, regularized = True))
    return res

def integrate(y, index, ms, sigmas):
    log_pdf = get_log_pdf(y, ms[index], sigmas[index])
    log_cdf = 0
    shape = ms.shape[0]
    for i in range(shape):
        if i != index:
            log_cdf += get_log_cdf(y, ms[i], sigmas[i])
    return mp.exp(log_pdf + log_cdf)

def beta_integrate(x, index, alphas, betas):
    log_pdf = get_log_beta_pdf(x, alphas[index], betas[index])
    log_cdf = 0
    shape = alphas.shape[0]
    for i in range(shape):
        if i != index:
            log_cdf += get_log_beta_cdf(x, alphas[i], betas[i])
    return mp.exp(log_pdf + log_cdf)


class Learner(ABC):
    """Base class of RL models implemented for the Mouselab-MDP paradigm."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def simulate(self, env, apply_microscope=False):
        pass
    
    # @abstractmethod
    # def init_model_params(self):
    #     pass

    def run_multiple_simulations(self, env, num_simulations,
                                 all_trials_data, compute_likelihood=False, apply_microscope=False):
        simulations_data = defaultdict(list)
        for _ in range(num_simulations):
            trials_data = self.simulate(
                env, all_trials_data=all_trials_data, compute_likelihood=compute_likelihood, apply_microscope=apply_microscope)
            simulations_data['r'].append(trials_data['r'])
            simulations_data['w'].append(trials_data['w'])
            simulations_data['a'].append(trials_data['a'])
            if 'loss' in trials_data:
                simulations_data['loss'].append(trials_data['loss'])
            if 'decision_params' in trials_data:
                simulations_data['decision_params'].append(
                    trials_data['decision_params'])
            if apply_microscope:
                simulations_data['s'].append(trials_data['s'])
        return dict(simulations_data)


class BaseRSSL(Learner):
    """Base class of the RSSL models with different priors"""
    # The total likelihood under the RSSL model would be the likelihood
    # of the strategy being selected and that multiplied with the
    # probability of the click sequence under the selected strategy
    # Change the source of the strategies.
    # Add participant temperatures
    def __init__(self, priors, strategy_space):
        super().__init__()
        self.priors = priors
        self.params = self.priors
        self.strategy_space = strategy_space
        self.num_strategies = len(strategy_space)
        self.upper_limit = 144
        self.lower_limit = -156
        self.gaussian = False
        # TODO:
        # Pass features and strategy weights to the model. For now, using a hack
        self.strategy_weights = pickle_load("data/microscope_weights.pkl")
        self.features = pickle_load("data/microscope_features.pkl")
        self.variance = 1 #Variance of the gaussian likelihood function
        self.action_log_probs = []

    def gaussian_max_likelihoods(self):
        params = self.params
        num_strategies = self.num_strategies
        means = params[:num_strategies]
        sigmas = np.sqrt(params[num_strategies:])
        max_val = np.max(means+5*sigmas)
        min_val = np.min(means-5*sigmas) 
        likelihoods = [mp.quad(lambda x: integrate(x, i, means, sigmas), [min_val, max_val]) for i in range(num_strategies)]
        return likelihoods
    
    def bernoulli_max_likelihoods(self):
        params = self.params
        num_strategies = self.num_strategies
        alphas = params[:num_strategies]
        betas = params[num_strategies:]
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
        params = self.params
        values = np.zeros(self.num_strategies)
        for strategy_num in range(self.num_strategies):
            values[strategy_num] = np.random.beta(
                params[strategy_num] + 1, params[strategy_num+self.num_strategies] + 1)
        return np.argmax(values)

    def gaussian_choice(self):
        params = self.params
        num_strategies = self.num_strategies
        values = np.zeros(num_strategies)
        for strategy_num in range(num_strategies):
            values[strategy_num] = np.random.normal(
                params[strategy_num], sqrt(params[strategy_num+num_strategies]))
        return np.argmax(values)

    def update_bernoulli_params(self, reward, strategy_index):
        num_strategies = self.num_strategies
        normalized_prob = (reward - self.lower_limit) / \
                (self.upper_limit-self.lower_limit)
        params = self.params
        choice = np.random.binomial(n=1, p=normalized_prob) == 1
        if choice:
            params[strategy_index] = params[strategy_index] + 1
        else:
            params[strategy_index+self.num_strategies] = params[strategy_index +
                                                                    self.num_strategies] + 1

    def update_gaussian_params(self, reward, strategy_index):
        var = self.variance
        num_strategies = self.num_strategies
        params = self.params
        params[strategy_index] = (params[strategy_index+num_strategies]*reward +
                                           params[strategy_index]*var)/(params[strategy_index+num_strategies] + var)
        params[strategy_index+num_strategies] = params[strategy_index +
                                                        num_strategies]*var/(params[strategy_index+num_strategies] + var)
    
    def update_params(self, reward, strategy_index):
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
        return actions, r_list

    def get_action_strategy_likelihood(self, trial, actions, chosen_strategy, temperature):
        strategy_weights =  self.strategy_weights[chosen_strategy-1]*(1/temperature)
        #Only for now
        normalized_features = get_normalized_features("high_increasing")
        ll = compute_log_likelihood(trial, actions, self.features, strategy_weights, inv_t=False, normalized_features=normalized_features)
        return ll

    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        env.reset()
        get_log_beta_cdf.cache_clear()
        get_log_beta_pdf.cache_clear()
        get_log_cdf.cache_clear()
        get_log_pdf.cache_clear()
        self.params = self.priors
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        temperature = all_trials_data['temperature']
        first_trial_data = {'actions': all_trials_data['actions'][0], 'rewards': all_trials_data['rewards'][0],
                                 'taken_path': all_trials_data['taken_paths'][0],
                                 'strategy': all_trials_data['strategies'][0]}
        for trial_num in range(num_trials):
            trial = env.trial_sequence.trial_sequence[trial_num]
            if compute_likelihood:
                clicks = all_trials_data['actions'][trial_num]
                rewards = all_trials_data['rewards'][trial_num]
                chosen_strategy = all_trials_data['strategies'][trial_num]
                likelihoods = self.get_max_likelihoods()
                strategy_index = self.strategy_space.index(chosen_strategy)
                strategy_likelihood = likelihoods[strategy_index]
                actions_strategy_log_likelihood = self.get_action_strategy_likelihood(trial, clicks, chosen_strategy, temperature)
                self.action_log_probs.append(float(str(actions_strategy_log_likelihood + mp.log(strategy_likelihood))))
                reward = np.sum(rewards)
                self.update_params(reward, strategy_index)
            else:
                strategy_index = self.select_strategy()
                clicks, r_list = self.apply_strategy(env, trial, strategy_index)
                reward = np.sum(r_list)
                # if trial_num == 0:
                #     reward = np.sum(first_trial_data['rewards'])
                #     chosen_strategy = first_trial_data['strategy']
                self.update_params(reward, strategy_index)
            trials_data['r'].append(reward)
            chosen_strategy = self.strategy_space[strategy_index] - 1
            trials_data['w'].append(self.strategy_weights[chosen_strategy])
            trials_data['a'].append(clicks)
            trials_data['s'].append(chosen_strategy+1)
            env.get_next_trial()
        
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)


class Policy(nn.Module):
    """ Softmax Policy of the REINFORCE model

        Implemented in PyTorch so that automatic gradients can be computed.
    """

    def __init__(self, beta, num_features):
        super(Policy, self).__init__()
        self.num_features = num_features
        # An array of dimension num_features
        self.weighted_preference = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=num_features, bias=False)
        self.beta = Variable(torch.tensor(beta), requires_grad=False)
        # as_a and as_b are parameters for the adaptive satisficing feature.
        self.as_a = nn.Parameter(torch.rand(1), requires_grad=True)
        self.as_b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.weighted_preference(x)
        action_scores = self.beta*x
        softmax_vals = F.log_softmax(action_scores, dim=0)
        softmax_vals = torch.exp(softmax_vals)
        return softmax_vals/softmax_vals.sum()


class BaseREINFORCE(Learner):
    """Base class of the REINFORCE model"""

    def __init__(self, lr, gamma, beta, features, init_weights, normalized_features=None,
                 use_pseudo_rewards=False, pr_weight=1):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.features = features
        self.num_features = len(features)
        self.num_actions = 13
        self.init_weights = np.array(init_weights)
        self.policy = Policy(self.beta, self.num_features).double()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.normalized_features = normalized_features
        self.use_pseudo_rewards = use_pseudo_rewards
        self.pr_weight = pr_weight
        self.delay_scale = 0
        self.init_model_params()
        self.action_log_probs = []

    def init_model_params(self):
        # Initializing the parameters with people's priors.
        self.policy.weighted_preference.weight.data = torch.DoubleTensor(
            [[self.init_weights*self.beta]])
        self.policy.as_a.data.fill_(0)
        self.policy.as_b.data.fill_(0)
    
    def get_term_reward(self, env):
        """Get the max expected return in the current state"""
        pres_node_map = env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    def get_action_details(self, env):
        """Generates action probabilities in the current state.

        Arguments:
            env {Gym env} -- Representation of the environment. 
        """
        available_actions = env.get_available_actions()
        present_node_map = env.present_trial.node_map
        mer = [present_node_map[action].calculate_max_expected_return(
        ) if action in available_actions else 0 for action in range(self.num_actions)]
        mer = torch.DoubleTensor(mer)
        X = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            X[action] = get_normalized_feature_values(
                present_node_map[action].compute_termination_feature_values(self.features), self.features, self.normalized_features)
        X = torch.DoubleTensor(X)
        # Normalize the adaptive satisficing feature
        if "num_clicks_adaptive" in self.features:
            as_index = self.features.index("num_clicks_adaptive")
            aspiration_value = autograd.Variable(
                mer - (self.policy.as_a - self.policy.as_b*X[:, as_index]))
            sig = nn.Sigmoid()
            if self.normalized_features:
                X[:, as_index] = (-sig(aspiration_value) + 1)/1
                X[0, as_index] = (0+1)/1
            else:
                X[:, as_index] = -sig(aspiration_value)
                X[0, as_index] = 0
        X = X.view(self.num_actions, 1, self.policy.num_features)
        available_actions = torch.LongTensor(available_actions)
        X_new = X[available_actions]
        probs = self.policy(X_new)
        complete_probs = torch.zeros(self.num_actions)
        for index, action in enumerate(available_actions):
            complete_probs[action] = probs[index]
        m = Categorical(complete_probs)
        return m

    def get_action(self, env):
        m = self.get_action_details(env)
        action = m.sample()
        # Saving log-action probabilities to compute gradients at episode end.
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def get_first_action(self, env, action):
        m = self.get_action_details(env)
        action = torch.tensor(action)
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        """Computing gradients and updating parameters.
        """
        R = 0
        policy_loss = []
        returns = []
        self.term_rewards.insert(0, 0)
        term_rewards = self.term_rewards[::-1]
        for i, r in enumerate(self.policy.rewards[::-1]):
            pr = self.pr_weight*(term_rewards[i] - term_rewards[i+1])
            R = (r+pr) + self.gamma*R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss)
        policy_loss = policy_loss.sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        return policy_loss.item()

    def update_reward_data(self, reward):
        self.policy.rewards.append(reward)

    def act_and_learn(self, env, new_action, reward, first_trial, first_action=False, first_path=None):
        """Only used in a two-stage model and serves as a message 
           passing function to perform a single action.

        Arguments:
            env {Gym env} -- Environment Representation
            new_action {int} -- Action to be taken.
            reward {float} -- Reward obtained when the decision stage selects a termination action.
            first_trial {bool} -- Whether the action is a first trial action.

        Keyword Arguments:
            first_action {bool} -- [Whether the action is first action of a trial] (default: {False})
            first_path {list} -- [Path taken at the end of first trial] (default: {None})
        """
        # TODO:
        # Add pseudo reward
        if first_trial:
            action = self.get_first_action(env, new_action)
            path = first_path
        else:
            action = self.get_action(env)
        term_reward = self.get_term_reward(env)
        self.term_rewards.append(term_reward)
        _, reward, done, info = env.step(action)
        if not first_trial:
            path = info
        self.update_reward_data(reward)
        if done:
            delay = env.present_trial.get_action_feedback(path)
            self.policy.rewards[-1] = reward - self.delay_scale*delay
            for log_prob in policy.saved_log_probs:
                self.action_log_probs.append(log_prob.cpu().numpy())
            self.finish_episode()
        return action, reward, done

    def get_current_weights(self):
        return torch.squeeze(self.policy.weighted_preference.weight.data).tolist() + \
            [self.beta] + [self.policy.as_a.data, self.policy.as_b.data]

    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        self.init_model_params()
        env.reset()
        self.first_trial_data = {'actions': all_trials_data['actions'][0], 'rewards': all_trials_data['rewards'][0],
                                 'taken_path': all_trials_data['taken_paths'][0],
                                 'strategy': all_trials_data['strategies'][0]}
        policy_loss = 0
        loss = []
        for trial_num in range(num_trials):
            self.term_rewards = []
            actions = []
            rewards = []
            trials_data['w'].append(self.get_current_weights())
            if compute_likelihood:
                trial_actions = all_trials_data['actions'][trial_num]
                trial_rewards = all_trials_data['rewards'][trial_num]
                for action, reward in zip(trial_actions, trial_rewards):
                    action = self.get_first_action(env, action)
                    actions.append(action)
                    term_reward = self.get_term_reward(env)
                    self.term_rewards.append(term_reward)
                    _, _, done, _ = env.step(action)
                    self.update_reward_data(reward)
                    rewards.append(reward)
                    if done:
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        taken_path = self.first_trial_data['taken_path']
                        delay = env.present_trial.get_action_feedback(
                            taken_path)
                        self.policy.rewards[-1] = reward - \
                            self.delay_scale*delay
                        loss.append(torch.sum(torch.stack(self.policy.saved_log_probs)))
                        break
            else:
                while True:
                    action = self.get_action(env)
                    actions.append(action)
                    term_reward = self.get_term_reward(env)
                    self.term_rewards.append(term_reward)
                    s, reward, done, info = env.step(action)
                    self.policy.rewards.append(reward)
                    rewards.append(reward)
                    if done:
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        taken_path = info
                        delay = env.present_trial.get_action_feedback(
                            taken_path)
                        self.policy.rewards[-1] = reward - \
                            self.delay_scale*delay
                        break
            env.get_next_trial()
            policy_loss += self.finish_episode()

        if apply_microscope:
            trials_data = get_strategy_sequences(env, trials_data)

        if loss:
            trials_data['loss'] = -torch.sum(torch.stack(loss)).data.cpu().numpy()
        else:
            trials_data['loss'] = None

        return dict(trials_data)

class BaseLVOC(Learner):
    """Base class of the LVOC model"""
    # TODO:
    # A version of LVOC that terminates only in the decision stage

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term=False):
        super().__init__()
        self.standard_dev = standard_dev
        self.num_samples = int(num_samples)
        self.features = features
        self.num_features = len(features)
        self.init_weights = init_weights
        self.normalized_features = normalized_features
        self.use_pseudo_rewards = use_pseudo_rewards
        self.pr_weight = pr_weight
        self.eps = eps
        self.no_term = no_term
        self.subjective_cost = 0
        self.delay_scale = 0
        self.vicarious_learning = False
        self.termination_value_known = False
        self.monte_carlo_updates = False
        self.init_model_params()
        self.term_rewards = []

    def init_model_params(self):
        """Initialize model parameters and initialize weights with participant priors"""
        self.mean = self.init_weights
        self.precision = np.diag([1/(self.standard_dev)**2]*self.num_features)
        self.gamma_a = 1
        self.gamma_b = 1
        self.action_log_probs = []

    def get_current_weights(self):
        return self.mean.tolist()

    def plot_norm_dists(self, means, sigmas, available_actions):
        plt.figure(figsize=(15, 9))
        num_actions = means.shape[0]
        for i in range(num_actions):
            d = norm(means[i], sigmas[i]**0.5)
            rvs = d.rvs(size=10000)
            sns.kdeplot(rvs, label=f'Action {available_actions[i]}')
        plt.legend()
        plt.show()

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
        self.mean, self.precision, self.gamma_a, self.gamma_b = estimate_bayes_glm(
            f, r, self.mean, self.precision, self.gamma_a, self.gamma_b)

    def get_term_features(self, env):
        """Get features of the termination action"""
        pres_node_map = env.present_trial.node_map
        term_features = get_normalized_feature_values(
            pres_node_map[0].compute_termination_feature_values(self.features, adaptive_satisficing={}), 
            self.features, self.normalized_features)
        return term_features

    def get_term_reward(self, env):
        """Get the max expected return in the current state"""
        pres_node_map = env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    def get_action_details(self, env):
        """Get the best action and its features in the given state"""
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        num_available_actions = len(available_actions)
        q = np.zeros(num_available_actions)
        feature_vals = np.zeros((self.num_actions, self.num_features))
        sampled_weights = self.sample_weights()
        for index, action in enumerate(available_actions):
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self.features, adaptive_satisficing={})
            computed_features = get_normalized_feature_values(
                computed_features, self.features, self.normalized_features)
            feature_vals[action] = computed_features
            q[index] = np.dot(sampled_weights, computed_features)
        if self.termination_value_known:
            term_reward = env.present_trial.node_map[0].calculate_max_expected_return(
            )
            q[0] = term_reward
        if self.no_term:
            best_index = break_ties_random(q[1:].tolist())
            available_actions.remove(0)
            best_action = available_actions[best_index]
        else:
            best_index = break_ties_random(q.tolist())
            best_action = available_actions[best_index]
        random_action = np.random.binomial(1, p=self.eps)
        if random_action:
            best_action = np.random.choice(available_actions)
        return best_action, feature_vals[best_action]

    def get_action(self, env):
        """Select the best action and store the action features"""
        action, f_vals = self.get_action_details(env)
        self.f_vals = f_vals
        return action

    def get_first_trial_action_details(self, env, given_action):
        """ Get the feature values of the action to be taken in the first trial.

        Arguments:
            env {Gym.Env} -- Representation of the environment
            given_action {int} -- Action taken in first trial
        """
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        feature_vals = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self.features, adaptive_satisficing={})
            computed_features = get_normalized_feature_values(
                computed_features, self.features, self.normalized_features)
            feature_vals[action] = computed_features
        return given_action, feature_vals[given_action]

    def perform_first_action_updates(self, env, reward, term_features, term_reward):
        self.update_rewards.append(reward - self.subjective_cost)
        if self.vicarious_learning:
            self.update_params(term_features, term_reward)

    def perform_action_updates(self, env, features, reward, term_features, term_reward):
        q = np.dot(self.mean, self.next_features)
        pr = self.get_term_reward(
            env) - self.pr_weight*self.term_rewards[-1] if self.use_pseudo_rewards else 0
        self.update_rewards.append(reward + pr - self.subjective_cost)
        self.update_params(features, q + reward + pr - self.subjective_cost)
        if self.vicarious_learning:
            self.update_params(term_features, term_reward)

    def perform_end_episode_updates(self, env, features, reward, taken_path):
        delay = env.present_trial.get_action_feedback(taken_path)
        self.update_params(features, reward - self.delay_scale*delay)
        self.update_rewards.append(reward - self.delay_scale*delay)
        self.perform_montecarlo_updates()

    def perform_montecarlo_updates(self):
        if self.monte_carlo_updates:
            for i in range(len(self.update_features) - 1):
                self.update_params(
                    self.update_features[i], np.sum(self.update_rewards[i:]))

    # This function is only used in conjunction with the hierarchical learner
    def act_and_learn(self, env, new_action, reward, first_trial, first_action=False, first_path=None):
        if not first_action:
            action = self.next_action
            features = self.next_features
        if first_trial or new_action == 0:
            self.next_action, self.next_features = self.get_first_trial_action_details(
                env, new_action)
        else:
            self.next_action, self.next_features = self.get_action_details(env)
        self.update_features.append(self.next_features)
        term_reward = self.get_term_reward(env)
        term_features = self.get_term_features(env)
        self.term_rewards.append(term_reward)
        if first_trial:
            s, _, done, info = env.step(new_action)
            taken_action = new_action
        else:
            s, reward, done, info = env.step(self.next_action)
            taken_action = self.next_action

        if new_action is not None and new_action == 0:
            if first_action:
                features = self.next_features
            self.perform_end_episode_updates(env, features, reward, info)
        else:
            if not done:
                if first_action:
                    self.perform_first_action_updates(
                        env, reward, term_features, term_reward)
                else:
                    self.perform_action_updates(
                        env, features, reward, term_features, term_reward)
            else:
                if first_trial:
                    self.perform_end_episode_updates(
                        env, self.next_features, reward, first_path)
                else:
                    self.perform_end_episode_updates(
                        env, self.next_features, reward, info)
        return taken_action, reward, done

    def store_action_likelihood(self, env, given_action):
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        if self.no_term:
            available_actions.remove(0)
        num_available_actions = len(available_actions)
        feature_vals = np.zeros((num_available_actions, self.num_features))
        dists = np.zeros((num_available_actions, 2))
        cov = np.linalg.inv(self.precision)
        for index, action in enumerate(available_actions):
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self.features, adaptive_satisficing={})
            computed_features = get_normalized_feature_values(
                computed_features, self.features, self.normalized_features)
            feature_vals[index] = computed_features
            computed_features = np.array(computed_features)
            dists[index][0] = np.dot(computed_features, self.mean)
            dists[index][1] = np.dot(
                np.dot(computed_features, cov), computed_features.T)
        means = dists[:, 0]
        sigmas = dists[:, 1]**0.5

        # Very important to select good bounds for proper sampling.
        max_val = np.max(means+5*sigmas)
        min_val = np.min(means-5*sigmas)

        if num_available_actions == 1:
            probs = [1.0]
        else:
            probs = np.array([mp.quad(lambda x: integrate(x, i, means, sigmas), [min_val, max_val]) for i in range(num_available_actions)])

        action_index = available_actions.index(given_action)
        selected_action_prob = probs[action_index]
        eps = self.eps
        log_prob = float(str(mp.log((1-eps)*selected_action_prob + eps*(1/num_available_actions))))
        self.action_log_probs.append(log_prob)
        return given_action, feature_vals[action_index]

    # This function is only used in conjunction with the hierarchical learner
    def store_hierarchical_action_likelihood(self, env, new_action, reward, first_trial, first_action=False, first_path=None):
        if not first_action:
            action = self.next_action
            features = self.next_features
        self.next_action, self.next_features = self.store_action_likelihood(
            env, new_action)
        self.update_features.append(self.next_features)
        term_reward = self.get_term_reward(env)
        term_features = self.get_term_features(env)
        self.term_rewards.append(term_reward)
        s, _, done, info = env.step(new_action)
        taken_action = new_action

        if new_action is not None and new_action == 0:
            if first_action:
                features = self.next_features
            self.perform_end_episode_updates(env, features, reward, info)
        else:
            if not done:
                if first_action:
                    self.perform_first_action_updates(
                        env, reward, term_features, term_reward)
                else:
                    self.perform_action_updates(
                        env, features, reward, term_features, term_reward)
            else:
                self.perform_end_episode_updates(
                    env, self.next_features, reward, first_path)
        return taken_action, reward, done

    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        self.init_model_params()
        env.reset()
        get_log_pdf.cache_clear()
        get_log_cdf.cache_clear()
        self.num_actions = len(env.get_available_actions())
        self.first_trial_data = {'actions': all_trials_data['actions'][0], 'rewards': all_trials_data['rewards'][0],
                                 'taken_path': all_trials_data['taken_paths'][0],
                                 'strategy': all_trials_data['strategies'][0]}
        self.all_trials_data = all_trials_data
        for trial_num in range(num_trials):
            self.update_rewards, self.update_features = [], []
            trials_data['w'].append(self.get_current_weights())
            actions, rewards, self.term_rewards = [], [], []
            if compute_likelihood:
                trial_actions = self.all_trials_data['actions'][trial_num]
                trial_rewards = self.all_trials_data['rewards'][trial_num]
                a_index, r_index = 0, 0
                action = trial_actions[a_index]
                a_index += 1
                next_action, self.next_features = self.store_action_likelihood(
                    env, action)
                actions.append(next_action)
                while True:
                    action = next_action
                    features = self.next_features
                    term_reward = self.get_term_reward(env)
                    term_features = self.get_term_features(env)
                    self.term_rewards.append(term_reward)
                    _, _, done, _ = env.step(action)
                    reward = trial_rewards[r_index]
                    r_index += 1
                    rewards.append(reward)
                    if not done:
                        action = trial_actions[a_index]
                        a_index += 1
                        # FIX THIS
                        next_action, self.next_features = self.store_action_likelihood(
                            env, action)
                        actions.append(next_action)
                        self.update_features.append(features)
                        self.perform_action_updates(
                            env, features, reward, term_features, term_reward)
                    else:
                        taken_path = self.all_trials_data['taken_paths'][trial_num]
                        self.update_features.append(features)
                        self.perform_end_episode_updates(
                            env, features, reward, taken_path)
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        env.get_next_trial()
                        break
            else:
                if trial_num == 0:
                    first_trial_actions = self.first_trial_data['actions']
                    first_trial_rewards = self.first_trial_data['rewards']
                    a_index, r_index = 0, 0
                    action = first_trial_actions[a_index]
                    a_index += 1
                    next_action, self.next_features = self.get_first_trial_action_details(
                        env, action)
                else:
                    next_action, self.next_features = self.get_action_details(
                        env)  # Is this correct?
                actions.append(next_action)
                while True:
                    action = next_action
                    features = self.next_features
                    term_reward = self.get_term_reward(env)
                    term_features = self.get_term_features(env)
                    self.term_rewards.append(term_reward)
                    if trial_num == 0:
                        _, _, done, _ = env.step(action)
                        reward = first_trial_rewards[r_index]
                        r_index += 1
                    else:
                        _, reward, done, info = env.step(action)
                    rewards.append(reward)
                    if not done:
                        if trial_num == 0:
                            action = first_trial_actions[a_index]
                            a_index += 1
                            next_action, self.next_features = self.get_first_trial_action_details(
                                env, action)
                        else:
                            next_action, self.next_features = self.get_action_details(
                                env)  # Is this correct?
                        actions.append(next_action)
                        self.update_features.append(features)
                        self.perform_action_updates(
                            env, features, reward, term_features, term_reward)
                    else:
                        if trial_num == 0:
                            taken_path = self.first_trial_data['taken_path']
                        else:
                            taken_path = info
                        self.update_features.append(features)
                        self.perform_end_episode_updates(
                            env, features, reward, taken_path)
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        env.get_next_trial()
                        break
        if apply_microscope:
            trials_data = get_strategy_sequences(env, trials_data)
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)

class DiscoverySelection(Learner):
    def __init__(self, standard_dev, num_samples, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        # TODO:
        # Implement delay scale if required

        super().__init__()
        self._standard_dev = standard_dev
        self._num_samples = num_samples
        self._features = features
        self._num_features = len(features)
        self._threshold = threshold
        self._bandit_params = np.array(bandit_params)
        self._num_strategies = int(self._bandit_params.shape[0]/2)
        if strategy_weights is not None:
            self._strategy_weights = np.array(strategy_weights)
        else:
            self._strategy_weights = np.random.rand(self._num_strategies, self._num_features)
        self._normalized_features = normalized_features
        self._threshold = threshold
        self._use_pr = use_pseudo_rewards
        self._pr_weight = pr_weight
        self._eps = eps
        self._upper_limit = 144
        self._lower_limit = -156
        self._subjective_cost = 0
        self._termination_value_known = False
        self._vicarious_learning = False
        self._monte_carlo_updates = False
        self._delay_scale = 0
        self.init_ts_params()

    def init_ts_params(self):
        self.mean = self._strategy_weights
        num_strategies = self._num_strategies
        self.precision = np.zeros((num_strategies, self._num_features, self._num_features))
        for s_num in range(num_strategies):
            self.precision[s_num] = np.diag([1/(self._standard_dev)**2]*self._num_features)
        self.gamma_a = np.ones(num_strategies)
        self.gamma_b = np.ones(num_strategies)
    
    def select_strategy(self):
        params = self._bandit_params
        num_strategies = self._num_strategies
        values = np.zeros(num_strategies)
        for strategy_num in range(num_strategies):
            values[strategy_num] = np.random.beta(
                params[strategy_num] + 1, params[strategy_num+num_strategies] + 1)
        return np.argmax(values)

    def sample_weights(self, s):
        """Sample weights from the posterior distribution"""
        sampled_weights = sample_coeffs(
            self.mean[s], self.precision[s], self.gamma_a[s], self.gamma_b[s], self._num_samples)
        return rows_mean(sampled_weights)

    def get_term_features(self, env):
        """Get features of the termination action"""
        pres_node_map = env.present_trial.node_map
        term_features = get_normalized_feature_values(
            pres_node_map[0].compute_termination_feature_values(self._features), 
            self._features, self._normalized_features)
        return term_features

    def get_term_reward(self, env):
        """Get the max expected return in the current state"""
        pres_node_map = env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        return term_reward

    def get_action_details(self, env, strategy_num):
        """Get the best action and its features in the given state"""
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        num_available_actions = len(available_actions)
        q = np.zeros(num_available_actions)
        feature_vals = np.zeros((self._num_actions, self._num_features))
        sampled_weights = self.sample_weights(strategy_num)
        for index, action in enumerate(available_actions):
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self._features)
            computed_features = get_normalized_feature_values(
                computed_features, self._features, self._normalized_features)
            feature_vals[action] = computed_features
            q[index] = np.dot(sampled_weights, computed_features)
        if self._termination_value_known:
            term_reward = env.present_trial.node_map[0].calculate_max_expected_return(
            )
            q[0] = term_reward
        best_index = break_ties_random(q.tolist())
        best_action = available_actions[best_index]
        random_action = np.random.binomial(1, p=self._eps)
        if random_action:
            best_action = np.random.choice(available_actions)
        return best_action, feature_vals[best_action]

    def update_bernoulli_params(self, reward, chosen_strategy):
        num_strategies = self._num_strategies
        normalized_reward = (reward - self._lower_limit) / \
                (self._upper_limit-self._lower_limit)
        params = self._bandit_params
        alpha = params[chosen_strategy]
        beta = params[chosen_strategy + num_strategies]
        params[chosen_strategy] += normalized_reward
        params[chosen_strategy + num_strategies] += (1-normalized_reward)
        C = self._threshold
        if alpha + beta >= C:
            params[chosen_strategy] *= (C/(C+1))
            params[chosen_strategy + num_strategies] *= (C/(C+1))
    
    def update_lvoc_params(self, strategy_num, f, r):
        s = strategy_num
        self.mean[s], self.precision[s], self.gamma_a[s], self.gamma_b[s] = estimate_bayes_glm(
                        f, r, self.mean[s], self.precision[s], self.gamma_a[s], self.gamma_b[s])


    def get_lvoc_details(self, env, strategy_num):
        """Select the best action and store the action features"""
        env.reset_trial()
        r_list = []
        actions = []
        episode_features = []
        episode_rewards = []
        term_rewards = [0]
        action, f_next = self.get_action_details(env, strategy_num)
        term_rewards.append(self.get_term_reward(env))
        episode_features.append(f_next)
        actions.append(action)
        while(1):
            f = f_next
            _, r, done, _ = env.step(action)
            r_list.append(r)
            update_r = r - self._subjective_cost
            episode_rewards.append(update_r)
            if not done:
                action, f_next = self.get_action_details(env, strategy_num)
                term_rewards.append(self.get_term_reward(env))
                episode_features.append(f_next)
                actions.append(action)
                q = np.dot(self.mean[strategy_num], f_next)
                update_r += q
            episode_features.append(f)
            pr = self._pr_weight*((term_rewards[-1] - term_rewards[-2]) if self._use_pr else 0)
            self.update_lvoc_params(strategy_num, f, update_r + pr)
            if done:
                if self._monte_carlo_updates:
                    for i in range(len(episode_features) - 1):
                        self.update_lvoc_params(strategy_num,
                            episode_features[i], np.sum(episode_rewards[i:]))
                break
            if self._vicarious_learning:
                term_features = self.get_term_features(env)
                term_reward = self.get_term_reward(env)
                self.update_lvoc_params(strategy_num, term_features, term_reward)
        return actions, r_list
    
    def apply_strategy(self, env, strategy_num):
        strategy_weights = self._strategy_weights[strategy_num]
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

    def get_action_features_likelihood(self, env, given_action, mean, precision):
        current_trial = env.present_trial
        available_actions = env.get_available_actions()
        num_available_actions = len(available_actions)
        feature_vals = np.zeros((num_available_actions, self._num_features))
        dists = np.zeros((num_available_actions, 2))
        cov = np.linalg.inv(precision)
        for index, action in enumerate(available_actions):
            computed_features = current_trial.node_map[action].compute_termination_feature_values(
                self._features)
            computed_features = get_normalized_feature_values(
                computed_features, self._features, self._normalized_features)
            feature_vals[index] = computed_features
            computed_features = np.array(computed_features)
            dists[index][0] = np.dot(computed_features, mean)
            dists[index][1] = np.dot(
                np.dot(computed_features, cov), computed_features.T)
        means = dists[:, 0]
        sigmas = dists[:, 1]**0.5
        if self._termination_value_known:
            term_reward = env.present_trial.node_map[0].calculate_max_expected_return(
            )
            dists[0][0] = term_reward
            dists[0][1] = 0
        max_val = np.max(means+5*sigmas)
        min_val = np.min(means-5*sigmas)
        if num_available_actions == 1:
            probs = [1.0]
        else:
            probs = np.array([mp.quad(lambda x: integrate(x, i, means, sigmas), [min_val, max_val]) for i in range(num_available_actions)])
        action_index = available_actions.index(given_action)
        selected_action_prob = probs[action_index]
        eps = self._eps
        log_prob = float(str(mp.log((1-eps)*selected_action_prob + eps*(1/num_available_actions))))
        return given_action, feature_vals[action_index], log_prob
    
    def get_pseudo_lvoc_log_likelihood(self, env, actions, rewards, strategy_num, true_update = False):
        env.reset_trial()
        log_likelihoods = []
        s = strategy_num
        episode_features = []
        episode_rewards = []
        term_rewards = [0]
        current_action_index = 0
        current_reward_index = 0
        if not true_update:
            mean = self.mean[s].copy()
            precision = self.precision[s].copy()
            gamma_a = self.gamma_a[s].copy()
            gamma_b = self.gamma_b[s].copy()
        given_action = actions[current_action_index]
        current_action_index += 1
        if true_update:
            action, f_next, log_prob = self.get_action_features_likelihood(env, given_action, self.mean[s], self.precision[s])
        else:
            action, f_next, log_prob = self.get_action_features_likelihood(env, given_action, mean, precision)
        term_rewards.append(self.get_term_reward(env))
        episode_features.append(f_next)
        log_likelihoods.append(log_prob)
        actions.append(action)
        while(1):
            f = f_next
            _, _, done, _ = env.step(action)
            r = rewards[current_reward_index]
            current_reward_index += 1
            update_r = r - self._subjective_cost
            episode_rewards.append(update_r)
            if not done:
                given_action = actions[current_action_index]
                current_action_index += 1
                if true_update:
                    action, f_next, log_prob = self.get_action_features_likelihood(env, given_action, self.mean[s], self.precision[s])
                else:
                    action, f_next, log_prob = self.get_action_features_likelihood(env, given_action, mean, precision)
                term_rewards.append(self.get_term_reward(env))
                episode_features.append(f_next)
                log_likelihoods.append(log_prob)
                actions.append(action)
                q = np.dot(self.mean[strategy_num], f_next)
                update_r += q
            pr = self._pr_weight*((term_rewards[-1] - term_rewards[-2]) if self._use_pr else 0)
            if not true_update:
                mean, precision, gamma_a, gamma_b = estimate_bayes_glm(f, update_r + pr, mean, precision, gamma_a, gamma_b)
            else:
                self.update_lvoc_params(strategy_num, f, update_r)
            if done:
                if self._monte_carlo_updates:
                    for i in range(len(episode_features) - 1):
                        if true_update:
                            self.update_lvoc_params(strategy_num,
                                episode_features[i], np.sum(episode_rewards[i:]))
                        else:
                            mean, precision, gamma_a, gamma_b = estimate_bayes_glm(episode_features[i], np.sum(episode_rewards[i:]),
                                                                mean, precision, gamma_a, gamma_b)
                break
            if self._vicarious_learning:
                term_features = self.get_term_features(env)
                term_reward = self.get_term_reward(env)
                if not true_update:
                    mean, precision, gamma_a, gamma_b = estimate_bayes_glm(term_features, term_reward, mean, precision, gamma_a, gamma_b)
                else:
                    self.update_lvoc_params(strategy_num, term_features, term_reward)
        return np.sum(log_likelihoods)
    
    def compute_log_likelihood(self, env, actions, rewards):
        params = self._bandit_params
        num_strategies = self._num_strategies
        alphas = params[:num_strategies]
        betas = params[num_strategies:]
        max_val = 1
        min_val = 0
        likelihoods = [mp.quad(lambda x: beta_integrate(x, i, alphas, betas), [min_val, max_val]) for i in range(num_strategies)]
        strategy_log_likelihoods = [mp.log(lik) for lik in likelihoods]
        lvoc_log_likelihoods = [self.get_pseudo_lvoc_log_likelihood(env, actions, rewards, i) for i in range(num_strategies)]
        # See if this is the right way to do things
        log_likelihoods = np.array(strategy_log_likelihoods, dtype = float) + np.array(lvoc_log_likelihoods, dtype = float)
        log_likelihood = logsumexp(log_likelihoods)
        return log_likelihood
    
    def simulate(self, env, all_trials_data = None, compute_likelihood = False, apply_microscope = False):
        env.reset()
        get_log_beta_cdf.cache_clear()
        get_log_beta_pdf.cache_clear()
        # self.init_ts_params()
        self.action_log_probs = []
        num_trials = env.num_trials
        trials_data = defaultdict(list)
        for trial_num in range(num_trials):
            self._num_actions = len(env.get_available_actions())
            if compute_likelihood:
                actions = all_trials_data['actions'][trial_num]
                rewards = all_trials_data['rewards'][trial_num]
                self.action_log_probs.append(self.compute_log_likelihood(env, actions, rewards))
                chosen_strategy = self.select_strategy()
                self.get_pseudo_lvoc_log_likelihood(env, actions, rewards, chosen_strategy, True)
                reward = np.sum(rewards)
                self.update_bernoulli_params(reward, chosen_strategy)
            else:
                chosen_strategy = self.select_strategy()
                actions, r_list = self.get_lvoc_details(env, chosen_strategy)
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

class HierarchicalAgent():
    """Agent that performs the decision to terminate or continue"""

    def __init__(self, tau, decision_rule='threshold', decision_params=None, features = None):
        self.tau = tau
        self.payoffs = []
        self.decision_rule = decision_rule
        self.decision_params = decision_params
        self.decision_params = self.get_init_decision_params(
        ) if not self.decision_params else decision_params
        self.previous_decision_params = get_zero_params(self.decision_params)
        self.features = features
        self.termination_features = [1, 2, 4, 18, 19,
                                     20, 21, 22, 23, 24, 27, 29, 30, 35, 44, 48]
        #self.normalize = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        #                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 72.0, 2.0, 2.0]
        self.max_payoff = 0
        self.avg_payoff = 0
        self.history = []
        self.action_log_probs = []

    def update_payoffs(self, total_reward):
        self.payoffs.append(total_reward)
        avg_payoff = rows_mean(self.payoffs)
        self.avg_payoff = avg_payoff
        if total_reward > self.max_payoff:
            self.max_payoff = total_reward

    def update_history(self, max_expected_return):
        self.history.append(max_expected_return)

    def init_model_params(self):
        self.payoffs = []
        self.max_payoff = 0
        self.avg_payoff = 0
        self.history = []

    def get_current_params(self):
        return self.decision_params.copy()

    def get_init_decision_params(self):
        decision_rule = self.decision_rule
        if decision_rule == "adaptive_satisficing":
            params = {'a': 1, 'b': 1}
        elif decision_rule in ["threshold", "best_path_difference"]:
            params = {'theta': 40}
        elif decision_rule in ["best_payoff", "average_payoff"]:
            params = {'theta': 0.5}
        elif decision_rule in ["VPI", "VOI1"]:
            params = {"theta": 30}
        elif decision_rule in ["maximum_improvement", "expected_improvement"]:
            params = {'theta': 10}
        elif decision_rule == "quantile":
            params = {'theta': 0.5}
        elif decision_rule == "noisy_memory_best_payoff":
            params = {'alpha': 1, 'beta': 1}
        return params

    def compute_stop_prob(self, env, max_expected_return=0, max_payoff=0, avg_payoff=0, vpi=0, voi=0, max_improvement=0, expected_improvement=0, path_history=[0]):
        decision_rule = self.decision_rule
        decision_params = self.decision_params
        tau = self.tau
        if len(env.get_available_actions()) == 1:
            return 1.0
        if decision_rule == "threshold":
            p_stop = temp_sigmoid(max_expected_return -
                                  decision_params['theta'], tau)
        elif decision_rule == "best_payoff":
            p_stop = temp_sigmoid(max_expected_return -
                                  decision_params['theta']*max_payoff, tau)
        elif decision_rule == "average_payoff":
            p_stop = temp_sigmoid(max_expected_return -
                                  decision_params['theta']*avg_payoff, tau)
        elif decision_rule == "adaptive_satisficing":
            num_clicks = 13 - len(env.get_available_actions())
            p_stop = temp_sigmoid(
                max_expected_return - decision_params['a'] + decision_params['b']*num_clicks, tau)
        elif decision_rule == "feature":
            # Add adaptive satisficing
            normalized_feature_values = env.present_trial.node_map[0].compute_termination_feature_values(
                self.features, adaptive_satisficing={})
            termination_feature_values = [normalized_feature_values[i] for i in self.termination_features]
            decision_weights = [self.decision_params[f"f_{i}"] for i in range(len(self.decision_params))]
            dot_product = np.dot(termination_feature_values, decision_weights)
            p_stop = temp_sigmoid(dot_product, tau)
        elif decision_rule == "best_path_difference":
            p_stop = temp_sigmoid(
                max_payoff - max_expected_return - decision_params['theta'], tau)
        elif decision_rule == "VPI":
            p_stop = temp_sigmoid(
                vpi - max_expected_return - decision_params['theta'], tau)
        elif decision_rule == "VOI1":
            p_stop = temp_sigmoid(
                voi - max_expected_return - decision_params['theta'], tau)
        elif decision_rule == "maximum_improvement":
            p_stop = temp_sigmoid(
                max_improvement - decision_params['theta'], tau)
        elif decision_rule == "expected_improvement":
            p_stop = temp_sigmoid(expected_improvement -
                                  decision_params['theta'], tau)
        elif decision_rule == "quantile":
            if decision_params['theta'] > 1:
                decision_params['theta'] = 1
            if decision_params['theta'] < 0:
                decision_params['theta'] = 0
            p_stop = temp_sigmoid(np.quantile(
                path_history, decision_params['theta']), tau)
        elif decision_rule == "noisy_memory_best_payoff":
            alpha = decision_params['alpha']
            beta = decision_params['beta']
            p_stop = 0
            trial_num = env.present_trial_num
            for reward in set(path_history):
                p_forget_higher = 1
                p_forget_reward = 1
                for i, higher_reward in enumerate(path_history):
                    delta = trial_num - i
                    if higher_reward > reward:
                        p_forget_higher *= sp.stats.gamma.pdf(
                            delta, a=alpha, scale=beta)
                    if higher_reward == reward:
                        p_forget_reward *= sp.stats.gamma.pdf(
                            delta, a=alpha, scale=beta)
                p_remember_reward = 1 - p_forget_reward
                p_stop_conditional = temp_sigmoid(
                    max_expected_return - decision_params['theta']*reward, tau)
                p_stop += p_remember_reward*p_forget_higher*p_stop_conditional
        elif decision_rule == "confidence_bound":
            trial_num = env.present_trial_num
            if trial_num == 0:
                threshold_mean = decision_params['threshold_mean']
            else:
                threshold_mean = self.avg_payoff
            p_stop = temp_sigmoid(max_expected_return - np.random.normal(
                threshold_mean, decision_params['threshold_var']/np.sqrt(trial_num + 1)), tau)
        return p_stop

    def get_action(self, env):
        max_expected_return = env.present_trial.node_map[0].calculate_max_expected_return(
        )
        # vpi = env.present_trial.node_map[0].calculate_vpi()
        # voi = env.present_trial.node_map[0].calculate_myopic_voi()
        # max_improv = env.present_trial.node_map[0].calculate_max_improvement()
        # expected_improv = env.present_trial.node_map[0].calculate_expected_improvement(
        # )
        self.update_history(max_expected_return)
        # Change this when you have fixed the vpi computation
        # p_stop = self.compute_stop_prob(env=env, max_expected_return=max_expected_return, max_payoff=self.max_payoff, avg_payoff=self.avg_payoff, vpi=vpi, voi=voi,
        #                                max_improvement=max_improv, expected_improvement=expected_improv, path_history=self.history)
        p_stop = self.compute_stop_prob(
            env, max_expected_return=max_expected_return, path_history=self.history)
        termination_choice = np.random.choice([0, 1], p=[p_stop, 1-p_stop])
        return termination_choice, p_stop


class HierarchicalLearner(Learner):
    """ Two stage model of decision making"""

    def __init__(self, features, init_weights, normalized_features=None, decision_rule=None, second_stage_learner=None, decision_params=None, no_term=False, **model_params):
        self.model_params = model_params
        self.actor = second_stage_learner
        self.decision_rule = decision_rule
        self.decision_params = decision_params
        self.second_stage_learner = second_stage_learner
        self.features = features
        self.init_weights = init_weights
        self.normalized_features = normalized_features
        self.no_term = no_term
        self.decision_agent = HierarchicalAgent(
            tau=self.model_params['tau'], decision_rule=self.decision_rule, decision_params=self.decision_params, features = self.features)
        self.model_params.pop('tau')
        self.second_stage_models = {'lvoc': LVOC, 'vicarious_lvoc': VicariousLVOC, 'montecarlo_lvoc': MonteCarloLVOC, 'termination_lvoc': TerminationLVOC, 'termination_montecarlo_lvoc': TerminationMonteCarloLVOC,
                                    'delay_lvoc': DelayLVOC, 'delay_reinforce': DelayREINFORCE, 'cost_lvoc': CostLVOC, 'reinforce': REINFORCE, 'cost_delay_lvoc': CostLVOCDelay, 'full_lvoc': FullLVOC,
                                    'cost_term_mc_lvoc': CostTermMCLVOC}
        self.actor_agent = self.second_stage_models[self.actor](features=self.features, init_weights=self.init_weights, normalized_features=self.normalized_features, no_term = self.no_term,
                                                                **self.model_params)

    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        self.all_trials_data = all_trials_data
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        self.actor_agent.init_model_params()
        self.decision_agent.init_model_params()
        env.reset()
        get_log_pdf.cache_clear()
        get_log_cdf.cache_clear()
        self.actor_agent.num_actions = len(env.get_available_actions())
        for trial_num in range(num_trials):
            self.actor_agent.trial_rewards = []
            actions = []
            rewards = []
            self.actor_agent.update_rewards = []
            self.actor_agent.update_features = []
            self.actor_agent.term_rewards = []
            trials_data['w'].append(self.actor_agent.get_current_weights())
            trials_data['decision_params'].append(
                self.decision_agent.get_current_params())
            if compute_likelihood:
                trial_actions = self.all_trials_data['actions'][trial_num]
                trial_rewards = self.all_trials_data['rewards'][trial_num]
                first_trial_path = self.all_trials_data['taken_paths'][trial_num]
                for i in range(len(trial_actions)):
                    action = trial_actions[i]
                    reward = trial_rewards[i]
                    # Maybe this is the reason of hierarchical learner not winning AIC
                    _, p_stop = self.decision_agent.get_action(env)
                    if action != 0:
                        action_prob = 1 - p_stop
                        if 1 - p_stop == 0:
                            action_prob = precision_epsilon
                    else:
                        action_prob = p_stop
                        if p_stop == 0:
                            action_prob = precision_epsilon

                    self.decision_agent.action_log_probs.append(
                        np.log(action_prob))
                    if i == 0:
                        first_action = True
                    else:
                        first_action = False
                    first_action = True if i == 0 else False
                    rewards.append(reward)
                    actions.append(action)
                    if not self.no_term or action != 0:
                        action, _, done = self.actor_agent.store_hierarchical_action_likelihood(
                            env, new_action=action, reward=reward, first_trial=True, first_action=first_action, first_path=first_trial_path)
            else:
                if trial_num == 0:
                    first_trial_actions = self.all_trials_data['actions'][trial_num]
                    first_trial_rewards = self.all_trials_data['rewards'][trial_num]
                    first_trial_path = self.all_trials_data['taken_paths'][trial_num]
                    for i in range(len(first_trial_actions)):
                        action = first_trial_actions[i]
                        reward = first_trial_rewards[i]
                        first_action = True if i == 0 else False
                        rewards.append(reward)
                        actions.append(action)
                        if not self.no_term or action != 0:
                            action, _, done = self.actor_agent.act_and_learn(
                                env, new_action=action, reward=reward, first_trial=True, first_action=first_action, first_path=first_trial_path)
                        else:
                            _, _, _, _ = env.step(0)
                else:
                    a_index = 0
                    while True:
                        termination_choice, _ = self.decision_agent.get_action(
                            env)
                        first_action = True if a_index == 0 else False
                        if termination_choice == 0:
                            if not self.no_term:
                                action, reward, done = self.actor_agent.act_and_learn(
                                    env, new_action=termination_choice, reward=None, first_trial=False, first_action=first_action)                               
                            else:
                                _, reward, done, _ = env.step(0)
                            a_index += 1
                            rewards.append(reward)
                            actions.append(termination_choice)
                            break
                        else:
                            action, reward, done = self.actor_agent.act_and_learn(
                                env, new_action=None, reward=None, first_trial=False, first_action=first_action)
                            a_index += 1
                            rewards.append(reward)
                            actions.append(action)
                            if done:
                                break
            env.get_next_trial()
            trials_data['a'].append(actions)
            trials_data['r'].append(np.sum(rewards))
        if apply_microscope:
            trials_data = get_strategy_sequences(env, trials_data)
        if self.decision_agent.action_log_probs and self.actor_agent.action_log_probs:
            trials_data['loss'] = -(np.sum(self.decision_agent.action_log_probs) +
                                    np.sum(self.actor_agent.action_log_probs))
        else:
            trials_data['loss'] = None
        return dict(trials_data)


class LVOC(BaseLVOC):
    """Vanilla LVOC"""

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)


class CostLVOC(BaseLVOC):
    """LVOC model with subjective cost"""

    def __init__(self, standard_dev, num_samples, subjective_cost, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.subjective_cost = subjective_cost


class DelayLVOC(BaseLVOC):
    """LVOC model with delay incorporated"""

    def __init__(self, standard_dev, num_samples, delay_scale, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.delay_scale = delay_scale


class CostLVOCDelay(BaseLVOC):
    """LVOC model with delay and subjective cost"""

    def __init__(self, standard_dev, num_samples, delay_scale, subjective_cost, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.subjective_cost = subjective_cost
        self.delay_scale = delay_scale


class VicariousLVOC(BaseLVOC):
    """LVOC model that learns about termination at each step"""

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.vicarious_learning = True


class TerminationLVOC(BaseLVOC):
    """LVOC model with value of termination given (Max expected return)"""

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.termination_value_known = True


class MonteCarloLVOC(BaseLVOC):
    """LVOC model that performs MonteCarlo updates at the end of the episode"""

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.monte_carlo_updates = True


class TerminationMonteCarloLVOC(BaseLVOC):
    """LVOC model with termination value known and performing MC updates"""

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.monte_carlo_updates = True
        self.termination_value_known = True


class CostTermMCLVOC(BaseLVOC):
    """LVOC with subjective cost, termination value known and MC updates"""

    def __init__(self, standard_dev, num_samples, subjective_cost, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.subjective_cost = subjective_cost
        self.monte_carlo_updates = True
        self.termination_value_known = True


class FullLVOC(BaseLVOC):
    """LVOC with all the features"""

    def __init__(self, standard_dev, num_samples, delay_scale, subjective_cost, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        self.monte_carlo_updates = True
        self.termination_value_known = True
        self.vicarious_learning = True
        self.subjective_cost = subjective_cost
        self.delay_scale = delay_scale

class CostDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, subjective_cost, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._subjective_cost = subjective_cost

class TerminationDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._termination_value_known = True

class VicariousDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._vicarious_learning = True

class MonteCarloDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._monte_carlo_updates = True

class TerminationMonteCarloDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._monte_carlo_updates = True
        self._termination_value_known = True

class CostTermMCDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""

    def __init__(self, standard_dev, num_samples, subjective_cost, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._subjective_cost = subjective_cost
        self._monte_carlo_updates = True
        self._termination_value_known = True

class FullDS(DiscoverySelection):
    """Discovery selection model with subjective cost"""
    # No delay scale for now

    def __init__(self, standard_dev, num_samples, subjective_cost, features, bandit_params, strategy_weights = None,
                threshold = 10, use_pseudo_rewards = False, pr_weight = 1, eps = 0, normalized_features = None):
        super().__init__(standard_dev, num_samples, features, bandit_params, strategy_weights,
                threshold, use_pseudo_rewards, pr_weight, eps, normalized_features)
        self._monte_carlo_updates = True
        self._termination_value_known = True
        self._vicarious_learning = True
        self._subjective_cost = subjective_cost

class REINFORCE(BaseREINFORCE):
    """Vanilla REINFORCE"""

    def __init__(self, lr, gamma, beta, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1):
        super().__init__(lr, gamma, beta, features,
                         init_weights, normalized_features, use_pseudo_rewards, pr_weight)


class DelayREINFORCE(BaseREINFORCE):
    """REINFORCE with delay incorporated"""

    def __init__(self, lr, gamma, beta, delay_scale, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1):
        super().__init__(lr, gamma, beta, features,
                         init_weights, normalized_features, use_pseudo_rewards, pr_weight)
        self.delay_scale = delay_scale


class BernoulliRSSL(BaseRSSL):
    """RSSL model with bernoulli priors"""

    def __init__(self, priors, strategy_space):
        super().__init__(priors, strategy_space)


class GaussianRSSL(BaseRSSL):
    """ RSSL model with Gaussian priors"""

    def __init__(self, priors, strategy_space):
        super().__init__(priors, strategy_space)
        self.gaussian = True

class NullLVOC(BaseLVOC):
    """ LVOC model with no learning """

    def __init__(self, standard_dev, num_samples, features, init_weights, normalized_features=None, use_pseudo_rewards=False, pr_weight=1, eps=0, no_term = False):
        super().__init__(standard_dev, num_samples, features, init_weights,
                         normalized_features, use_pseudo_rewards, pr_weight, eps, no_term)
        
    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        self.init_model_params()
        env.reset()
        get_log_pdf.cache_clear()
        get_log_cdf.cache_clear()
        self.num_actions = len(env.get_available_actions())
        self.first_trial_data = {'actions': all_trials_data['actions'][0], 'rewards': all_trials_data['rewards'][0],
                                'taken_path': all_trials_data['taken_paths'][0],
                                'strategy': all_trials_data['strategies'][0]}
        self.all_trials_data = all_trials_data
        for trial_num in range(num_trials):
            trials_data['w'].append(self.get_current_weights())
            actions, rewards, self.term_rewards = [], [], []
            if compute_likelihood:
                trial_actions = self.all_trials_data['actions'][trial_num]
                trial_rewards = self.all_trials_data['rewards'][trial_num]
                a_index, r_index = 0, 0
                action = trial_actions[a_index]
                a_index += 1
                next_action, self.next_features = self.store_action_likelihood(
                    env, action)
                actions.append(next_action)
                while True:
                    action = next_action
                    features = self.next_features
                    term_reward = self.get_term_reward(env)
                    term_features = self.get_term_features(env)
                    self.term_rewards.append(term_reward)
                    _, _, done, _ = env.step(action)
                    reward = trial_rewards[r_index]
                    r_index += 1
                    rewards.append(reward)
                    if not done:
                        action = trial_actions[a_index]
                        a_index += 1
                        # FIX THIS
                        next_action, self.next_features = self.store_action_likelihood(
                            env, action)
                        actions.append(next_action)
                    else:
                        taken_path = self.all_trials_data['taken_paths'][trial_num]
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        env.get_next_trial()
                        break
            else:
                if trial_num == 0:
                    first_trial_actions = self.first_trial_data['actions']
                    first_trial_rewards = self.first_trial_data['rewards']
                    a_index, r_index = 0, 0
                    action = first_trial_actions[a_index]
                    a_index += 1
                    next_action, self.next_features = self.get_first_trial_action_details(
                        env, action)
                else:
                    next_action, self.next_features = self.get_action_details(
                        env)  # Is this correct?
                actions.append(next_action)
                while True:
                    action = next_action
                    features = self.next_features
                    term_reward = self.get_term_reward(env)
                    term_features = self.get_term_features(env)
                    self.term_rewards.append(term_reward)
                    if trial_num == 0:
                        _, _, done, _ = env.step(action)
                        reward = first_trial_rewards[r_index]
                        r_index += 1
                    else:
                        _, reward, done, info = env.step(action)
                    rewards.append(reward)
                    if not done:
                        if trial_num == 0:
                            action = first_trial_actions[a_index]
                            a_index += 1
                            next_action, self.next_features = self.get_first_trial_action_details(
                                env, action)
                        else:
                            next_action, self.next_features = self.get_action_details(
                                env)  # Is this correct?
                        actions.append(next_action)
                    else:
                        if trial_num == 0:
                            taken_path = self.first_trial_data['taken_path']
                        else:
                            taken_path = info
                        trials_data['r'].append(np.sum(rewards))
                        trials_data['a'].append(actions)
                        env.get_next_trial()
                        break
        if apply_microscope:
            trials_data = get_strategy_sequences(env, trials_data)
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)

class NullRSSL(BaseRSSL):
    def __init__(self, priors, strategy_space):
        super().__init__(priors, strategy_space)
    
    def simulate(self, env, all_trials_data, apply_microscope=False, compute_likelihood=False):
        env.reset()
        get_log_beta_cdf.cache_clear()
        get_log_beta_pdf.cache_clear()
        get_log_cdf.cache_clear()
        get_log_pdf.cache_clear()
        self.params = self.priors
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        temperature = all_trials_data['temperature']
        first_trial_data = {'actions': all_trials_data['actions'][0], 'rewards': all_trials_data['rewards'][0],
                                 'taken_path': all_trials_data['taken_paths'][0],
                                 'strategy': all_trials_data['strategies'][0]}
        for trial_num in range(num_trials):
            trial = env.trial_sequence.trial_sequence[trial_num]
            if compute_likelihood:
                clicks = all_trials_data['actions'][trial_num]
                rewards = all_trials_data['rewards'][trial_num]
                chosen_strategy = all_trials_data['strategies'][trial_num]
                likelihoods = self.get_max_likelihoods()
                strategy_index = self.strategy_space.index(chosen_strategy)
                strategy_likelihood = likelihoods[strategy_index]
                actions_strategy_log_likelihood = self.get_action_strategy_likelihood(trial, clicks, chosen_strategy, temperature) #Fix this
                self.action_log_probs.append(float(str(actions_strategy_log_likelihood + mp.log(strategy_likelihood))))
                reward = np.sum(rewards)
            else:
                strategy_index = self.select_strategy()
                clicks, r_list = self.apply_strategy(env, trial, strategy_index)
                reward = np.sum(r_list)
            trials_data['r'].append(reward)
            chosen_strategy = self.strategy_space[strategy_index] - 1
            trials_data['w'].append(self.strategy_weights[chosen_strategy])
            trials_data['a'].append(clicks)
            trials_data['s'].append(chosen_strategy+1)
            env.get_next_trial()
        if self.action_log_probs:
            trials_data['loss'] = -np.sum(self.action_log_probs)
        else:
            trials_data['loss'] = None
        return dict(trials_data)

class NullBernoulliRSSL(NullRSSL):

    def __init__(self, priors, strategy_space):
        super().__init__(priors, strategy_space)

class NullGaussianRSSL(NullRSSL):
    
    def __init__(self, priors, strategy_space):
        super().__init__(priors, strategy_space)
        self.gaussian = True

class ValuePolicy(nn.Module):
    def __init__(self, num_features, num_actions):
        super(ValuePolicy, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.weighted_preference = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=num_features)
        self.linear = nn.Linear(num_actions, 1)
        self.baselines = []

    def init_weights(self):
        pass
    
    def forward(self, x):
        w_pref = self.weighted_preference(x).reshape(1, 1, -1)
        res = self.linear(w_pref)
        return res

class BaselineREINFORCE(BaseREINFORCE):
    """Base class of the REINFORCE model"""

    def __init__(self, lr, value_lr, gamma, beta, features, init_weights, normalized_features=None,
                 use_pseudo_rewards=False, pr_weight=1):
        super().__init__(lr, gamma, beta, features,
                         init_weights, normalized_features, use_pseudo_rewards, pr_weight)
        self.value_lr = value_lr
        self.value_policy = ValuePolicy(self.num_features, self.num_actions).double()
        self.value_policy.weighted_preference.weight.data = torch.zeros_like(self.value_policy.weighted_preference.weight, requires_grad=True)
        self.value_optimizer = optim.Adam(self.value_policy.parameters(), lr = self.value_lr)

    def get_action_details(self, env):
        """Generates action probabilities in the current state.

        Arguments:
            env {Gym env} -- Representation of the environment. 
        """
        available_actions = env.get_available_actions()
        present_node_map = env.present_trial.node_map
        mer = [present_node_map[action].calculate_max_expected_return(
        ) if action in available_actions else 0 for action in range(self.num_actions)]
        mer = torch.DoubleTensor(mer)
        X = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            X[action] = get_normalized_feature_values(
                present_node_map[action].compute_termination_feature_values(self.features), self.features, self.normalized_features)
        X = torch.DoubleTensor(X)
        # Normalize the adaptive satisficing feature
        if "num_clicks_adaptive" in self.features:
            as_index = self.features.index("num_clicks_adaptive")
            aspiration_value = autograd.Variable(
                mer - (self.policy.as_a - self.policy.as_b*X[:, as_index]))
            sig = nn.Sigmoid()
            if self.normalized_features:
                X[:, as_index] = (-sig(aspiration_value) + 1)/1
                X[0, as_index] = (0+1)/1
            else:
                X[:, as_index] = -sig(aspiration_value)
                X[0, as_index] = 0
        X = X.view(self.num_actions, 1, self.policy.num_features)
        available_actions = torch.LongTensor(available_actions)
        X_new = X[available_actions]
        probs = self.policy(X_new)
        complete_probs = torch.zeros(self.num_actions)
        for index, action in enumerate(available_actions):
            complete_probs[action] = probs[index]
        m = Categorical(complete_probs)
        baseline = self.value_policy(X)
        return m, baseline

    def get_action(self, env):
        m, baseline = self.get_action_details(env)
        action = m.sample()
        # Saving log-action probabilities to compute gradients at episode end.
        self.policy.saved_log_probs.append(m.log_prob(action))
        self.value_policy.baselines.append(baseline)
        return action.item()

    def get_first_action(self, env, action):
        m, baseline = self.get_action_details(env)
        action = torch.tensor(action)
        self.policy.saved_log_probs.append(m.log_prob(action))
        self.value_policy.baselines.append(baseline)
        return action.item()

    def finish_episode(self):
        """Computing gradients and updating parameters.
        """
        R = 0
        policy_loss = []
        value_loss = []
        returns = []
        self.term_rewards.insert(0, 0)
        term_rewards = self.term_rewards[::-1]
        for i, r in enumerate(self.policy.rewards[::-1]):
            pr = self.pr_weight*(term_rewards[i] - term_rewards[i+1])
            R = (r+pr) + self.gamma*R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        baselines = torch.tensor(self.value_policy.baselines[::-1], requires_grad=True).float()
        for log_prob, R, b in zip(self.policy.saved_log_probs, returns, baselines):
            policy_loss.append(-log_prob.float() * (R-b))
        policy_loss = torch.stack(policy_loss)
        policy_loss = policy_loss.sum().float()
        value_loss = -(baselines.sum()).float()
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.optimizer.step()
        self.value_optimizer.step()
        self.value_policy.baselines = []
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        return policy_loss.item()


models = {'lvoc': LVOC, 'vicarious_lvoc': VicariousLVOC, 'montecarlo_lvoc': MonteCarloLVOC,
          'termination_lvoc': TerminationLVOC, 'termination_montecarlo_lvoc': TerminationMonteCarloLVOC,
          'delay_lvoc': DelayLVOC, 'delay_reinforce': DelayREINFORCE, 'cost_lvoc': CostLVOC,
          'reinforce': REINFORCE, 'cost_delay_lvoc': CostLVOCDelay, 'hierarchical_learner': HierarchicalLearner,
          'full_lvoc': FullLVOC, 'bernoulli_rssl': BernoulliRSSL, 'gaussian_rssl': GaussianRSSL,
          'cost_term_mc_lvoc': CostTermMCLVOC, 'ds': DiscoverySelection,
          "vicarious_ds": VicariousDS, "cost_ds": CostDS, "montecarlo_ds": MonteCarloDS,
          "termination_ds": TerminationDS, "termination_montecarlo_ds": TerminationMonteCarloDS, "full_ds": FullDS,
          "null_lvoc": NullLVOC, "null_bernoulli_rssl": NullBernoulliRSSL, "null_gaussian_rssl": NullGaussianRSSL,
          "baseline_reinforce": BaselineREINFORCE}


class ParameterOptimizer():
    """Class to optimize the parameters of the models"""
    # TODO:
    # Update the class for discovery selection models
    def __init__(self, model_name, participant, all_trials_data, features, pipeline,
                 normalized_features=None, use_pseudo_rewards=False,
                 optimization_criterion="performance_error", num_simulations=30, max_evals=100,
                 decision_rule=None, second_stage_learner=None, optimize_priors=False, optimize_pr_weight=False,
                 no_term = False, strategy_space=None,
                 **params):
        self.model_name = model_name.lower()
        self.features = features
        self.participant = participant
        self.decision_rule = decision_rule
        self.decision_param_names = None
        self.second_stage_learner = second_stage_learner
        self.pipeline = pipeline
        self.env = GenericMouselabEnv(len(self.participant.envs), pipeline=self.pipeline,
                                      ground_truth=self.participant.envs)
        self.max_evals = max_evals
        self.num_simulations = num_simulations
        self.normalized_features = normalized_features
        self.use_pseudo_rewards = use_pseudo_rewards
        self.optimization_criterion = optimization_criterion
        self.all_trials_data = all_trials_data
        self.optimize_priors = optimize_priors
        self.optimize_pr_weight = optimize_pr_weight
        self.no_term = no_term
        self.strategy_space = strategy_space
        self.models = models
        self.parameter_ranges = self.get_parameter_ranges()

    def get_hyperparameter_range(self, decision_rule):
        """Getting search spaces of parameters according to the decision rule

        Arguments:
            decision_rule {str} -- Type of decision rule in the first stage
                                   of the two stage model
        """
        hp_space = {'tau': hp.loguniform('tau', np.log(1e-3), np.log(1e3))}
        if decision_rule == "adaptive_satisficing":
            hp_space['a'] = hp.loguniform('a', np.log(1e-2), np.log(1e2))
            hp_space['b'] = hp.loguniform('b', np.log(1e-2), np.log(1e2))
        elif decision_rule == "threshold":
            hp_space['theta'] = hp.uniform('theta', -150, 150)
        elif decision_rule in ["best_payoff", "average_payoff"]:
            hp_space['theta'] = hp.loguniform(
                'theta', np.log(0.01), np.log(1e2))
        elif decision_rule == "best_path_difference":
            hp_space['theta'] = hp.uniform('theta', -150, 150)
        elif decision_rule == "noisy_memory_best_payoff":
            hp_space['alpha'] = hp.uniform('alpha', 1, 10)
            hp_space['beta'] = hp.uniform('beta', 1, 5)
            hp_space['theta'] = hp.loguniform(
                'theta', np.log(0.01), np.log(1e2))
        elif decision_rule == "confidence_bound":
            hp_space['threshold_mean'] = hp.uniform('threshold_mean', -15, 55)
            hp_space['threshold_var'] = hp.uniform('threshold_var', 1, 100)
        elif decision_rule == "feature":
            for i in range(16):
                hp_space[f"f_{i}"] = hp.uniform(f'f_{i}', 0, 1)
        return hp_space

    def get_objective_value(self, simulations_data):
        """Compute the objective value to be minimized based on the optimization
           criterion and the data obtained by running the model with a given set
           of parameters

        Arguments:
            simulations_data {dict} -- Data from runs of models with a particular
                                       parameter configuration
        """
        criterion = self.optimization_criterion
        if criterion == "reward":
            objective_value = -np.mean(simulations_data['r'])
        elif criterion == "performance_error":
            objective_value = get_squared_performance_error(
                self.participant.scores, simulations_data['r'])
        elif criterion == "distance":
            objective_value = get_normalized_weight_distance(
                self.participant.weights, simulations_data['w'])
        elif criterion == "strategy_accuracy":
            p_s = {self.participant.pid: self.participant.strategies}
            a_s = {self.participant.pid: simulations_data['s']}
            objective_value = - \
                strategy_accuracy(p_s, a_s)[self.participant.pid]
        elif criterion == "strategy_transition":
            p_s = [[self.participant.strategies]]
            a_s = [simulations_data['s']]
            objective_value = compute_transition_distance(p_s, a_s)
        elif criterion == "clicks_overlap":
            p_a = {self.participant.pid: self.participant.clicks}
            a_a = {self.participant.pid: simulations_data['a']}
            objective_value = -clicks_overlap(p_a, a_a)[self.participant.pid]
        elif criterion == "likelihood":
            objective_value =  np.mean(simulations_data['loss'])
        print(objective_value)
        return objective_value

    def get_parameter_ranges(self):
        """Getting search spaces for parameters of the models"""
        param_ranges = {
        'standard_dev': hp.loguniform('standard_dev', np.log(1e-3), np.log(1e3)),
        'num_samples': hp.quniform('num_samples', 1, 10, 1),
        'delay_scale': hp.loguniform('delay_scale', np.log(1e-2), np.log(5)),
        'subjective_cost': hp.loguniform('subjective_cost', np.log(1e-2), np.log(60)),
        'lr': hp.loguniform('lr', np.log(1e-9), np.log(1e-1)),
        'beta': hp.loguniform('beta', np.log(1e-3), np.log(1e3)),
        'gamma': hp.loguniform('gamma', np.log(0.90), np.log(1-1e-6)),
        'threshold': hp.uniform('threshold', 1, 50),
        'num_strategies': hp.quniform('num_strategies', 1, 10, 1),
        'eps': hp.uniform('eps', 1e-3, 1),
        'value_lr': hp.loguniform('value_lr', np.log(1e-10), np.log(1e-1))
        }
        model = self.models[self.model_name]
        model_name = self.model_name
        model_args = inspect.getfullargspec(model.__init__).args
        total_param_ranges = {}
        if "rssl" in model_name:
            num_strategies = len(self.strategy_space)
            print(self.model_name)
            if "gaussian_rssl" in self.model_name:
                for i in range(num_strategies):
                    total_param_ranges[f'priors_{i}'] = hp.loguniform(f'priors_{i}', -3, 0)
                for i in range(num_strategies, 2*num_strategies):
                    total_param_ranges[f'priors_{i}'] = hp.loguniform(f'priors_{i}', -3, 3)
            else:
                for i in range(num_strategies):
                    total_param_ranges[f'priors_{i}'] = hp.loguniform(f'priors_{i}', 0, np.log(10))
                for i in range(num_strategies, 2*num_strategies):
                    total_param_ranges[f'priors_{i}'] = hp.loguniform(f'priors_{i}', 0, np.log(10))
        else:
            total_args = model_args
            if self.second_stage_learner:
                first_stage_params = self.get_hyperparameter_range(self.decision_rule)
                second_stage_args = inspect.getfullargspec(self.models[self.second_stage_learner].__init__).args
                total_args += second_stage_args
                self.decision_param_names = list(first_stage_params.keys())
                self.decision_param_names.remove('tau')
                total_param_ranges.update(first_stage_params)
            if self.optimize_pr_weight:
                total_param_ranges['pr_weight'] = hp.uniform('pr_weight', -5, 5)
            for arg in total_args:
                if arg in param_ranges:
                    total_param_ranges[arg] = param_ranges[arg]
            second_stage_learner = "none"
            if self.second_stage_learner:
                second_stage_learner = self.second_stage_learner
            if ("lvoc" in second_stage_learner[-4:] or "lvoc" in model_name[-4:]) or "reinforce" in model_name:
                if self.optimize_priors:
                    for i in range(len(self.features)):
                        total_param_ranges[f'w_{i}'] = hp.normal(f'w_{i}', 0, 1)
            if (model_name[-2:] == "ds"):
                total_param_ranges['num_strategies'] = param_ranges['num_strategies']
                total_param_ranges['alpha'] = hp.uniform('alpha', 1, 10)
                total_param_ranges['beta'] = hp.uniform('beta', 1, 10)
        return total_param_ranges

    def model_objective(self, params):
        """Call the right objective function based on the model type

        Arguments:
            params {dict} -- Parameters of the model
        """
        model = self.models[self.model_name]
        if self.optimize_priors:
            num_w_params = len(self.features)
            init_weights = np.zeros(num_w_params)
            for i in range(num_w_params):
                init_weights[i] = params[f'w_{i}']
                del params[f'w_{i}']
        else:
            init_weights = self.participant.weights[0]
        if "rssl" in self.model_name:
            priors = np.array([params[f'priors_{i}'] for i in range(len(params))])
            agent = model(priors=priors, strategy_space=self.strategy_space)
        elif self.model_name[-2:] == "ds":
            num_strategies = int(params['num_strategies'])
            num_samples = int(params['num_samples'])
            alpha = params['alpha']
            beta = params['beta']
            del params['num_samples'], params['num_strategies'], params['alpha'], params['beta']
            bandit_params = np.ones(2*num_strategies)
            bandit_params[:num_strategies] = alpha*bandit_params[:num_strategies]
            bandit_params[num_strategies:] = beta*bandit_params[num_strategies:]
            agent = model(num_samples = num_samples, use_pseudo_rewards = self.use_pseudo_rewards, 
                        features = self.features, bandit_params = bandit_params, strategy_weights = np.zeros((num_strategies, len(self.features))),
                        normalized_features = self.normalized_features, **params)
        else:
            if self.decision_param_names:
                self.decision_params = {k: params[k] for k in self.decision_param_names}
                dict_keys = list(self.decision_params.keys())
                for k in dict_keys:
                    params.pop(k)
                agent = model(features=self.features, init_weights=init_weights, normalized_features=self.normalized_features, use_pseudo_rewards=self.use_pseudo_rewards,
                              decision_rule=self.decision_rule, second_stage_learner=self.second_stage_learner, decision_params=self.decision_params, no_term = self.no_term, **params)
            else:
                agent = model(features=self.features, init_weights=init_weights, normalized_features=self.normalized_features, use_pseudo_rewards=self.use_pseudo_rewards,
                              **params)

        if self.optimization_criterion not in ["strategy_accuracy", "strategy_transition", "likelihood"]:
            simulations_data = agent.run_multiple_simulations(
                self.env, self.num_simulations, all_trials_data=self.all_trials_data)
        elif self.optimization_criterion == "likelihood":
            simulations_data = agent.run_multiple_simulations(
                self.env, self.num_simulations, all_trials_data=self.all_trials_data, compute_likelihood=True)
        else:
            simulations_data = agent.run_multiple_simulations(
                self.env, self.num_simulations, all_trials_data=self.all_trials_data, apply_microscope=True)
        objective_value = self.get_objective_value(simulations_data)
        return objective_value

    def get_best_parameters(self, trials=False):
        """Optimize the parameters for the selected model

        Keyword Arguments:
            trials {bool} -- Whether or not to store the results of function evaluations
                            (default: {False})
        """
        algo = partial(tpe.suggest, n_startup_jobs=30)
        #algo = hyperopt.rand.suggest
        if trials:
            trials = Trials()
            best_params = fmin(fn=self.model_objective, space=self.parameter_ranges,
                                algo=algo, max_evals=self.max_evals, trials=trials)
            return best_params, trials
        else:
            return fmin(fn=self.model_objective, space=self.parameter_ranges, algo=algo, max_evals=self.max_evals), {}


class ModelRunner():
    """Run the models with a given set of parameters"""
    # TODO:
    # Always runs with all trials

    def __init__(self, model_name, participant, features,
                 pipeline,
                 normalized_features=None, use_pseudo_rewards=False,
                 decision_rule=None, second_stage_learner=None, decision_params=None,
                 include_all_trials=True, no_term = False, strategy_space=None):
        self.model_name = model_name
        self.features = features
        self.decision_rule = decision_rule
        self.decision_params = decision_params
        self.second_stage_learner = second_stage_learner
        self.normalized_features = normalized_features
        self.use_pseudo_rewards = use_pseudo_rewards
        self.no_term = no_term
        self.strategy_space = strategy_space
        self.participant = participant
        self.pipeline = pipeline
        self.env = GenericMouselabEnv(len(self.participant.envs), pipeline=self.pipeline,
                                      ground_truth=self.participant.envs)

    def get_init_decision_params(self):
        decision_rule = self.decision_rule
        if decision_rule == "adaptive_satisficing":
            params = {'a': 1, 'b': 1}
        elif decision_rule in ["threshold", "best_path_difference"]:
            params = {'theta': 40}
        elif decision_rule in ["best_payoff", "average_payoff"]:
            params = {'theta': 0.5}
        elif decision_rule in ["VPI", "VOI1"]:
            params = {"theta": 30}
        elif decision_rule in ["maximum_improvement", "expected_improvement"]:
            params = {'theta': 10}
        elif decision_rule == "quantile":
            params = {'theta': 0.5}
        elif decision_rule == "noisy_memory_best_payoff":
            params = {'alpha': 1, 'beta': 1, 'theta': 0.5}
        elif decision_rule == "confidence_bound":
            params = {'threshold_mean': 40, 'threshold_var': 30}
        elif decision_rule == "feature":
            params = {}
            for i in range(16):
                params[f"f_{i}"] = 0.5
        return params

    def run(self, parameters, all_trials_data, num_simulations=30, apply_microscope=True, compute_likelihood=False):
        self.all_trials_data = all_trials_data
        params = parameters.copy()
        if 'w_0' in params:
            num_w = len(self.features)
            self.init_weights = np.zeros(num_w)
            for i in range(num_w):
                self.init_weights[i] = params[f'w_{i}']
                del params[f'w_{i}']
        else:
            self.init_weights = self.participant.weights[0]
        self.models = models
        model = self.models[self.model_name]
        copy_params = params.copy()
        if "rssl" in self.model_name:
            priors = np.array([params[f'priors_{i}'] for i in range(len(params))])
            agent = model(priors=priors, strategy_space=self.strategy_space)
        elif self.model_name[-2:] == "ds":
            num_strategies = int(params['num_strategies'])
            num_samples = int(params['num_samples'])
            alpha = params['alpha']
            beta = params['beta']
            del params['num_samples'], params['num_strategies'], params['alpha'], params['beta']
            bandit_params = np.ones(2*num_strategies)
            bandit_params[:num_strategies] = alpha*bandit_params[:num_strategies]
            bandit_params[num_strategies:] - beta*bandit_params[num_strategies:]
            agent = model(num_samples = num_samples, use_pseudo_rewards = self.use_pseudo_rewards, 
                        features = self.features, bandit_params = bandit_params, strategy_weights = np.zeros((num_strategies, len(self.features))),
                        normalized_features = self.normalized_features, **params)
        else:
            if self.decision_rule:
                for p in ['decision_rule', 'second_stage_learner']:
                    if p in copy_params:
                        copy_params.pop(p)
                init_decision_params = self.get_init_decision_params()
                self.decision_params = {k: params[k]
                                        for k in init_decision_params.keys()}
                decision_params_names = list(init_decision_params.keys())
                for dp in decision_params_names:
                    copy_params.pop(dp)
                agent = model(features=self.features, init_weights=self.init_weights, normalized_features=self.normalized_features, use_pseudo_rewards=self.use_pseudo_rewards, decision_rule=self.decision_rule, second_stage_learner=self.second_stage_learner,
                              decision_params=self.decision_params, no_term = self.no_term, **copy_params)
            else:
                agent = model(features=self.features, init_weights=self.init_weights,
                              normalized_features=self.normalized_features, use_pseudo_rewards=self.use_pseudo_rewards, **copy_params)
        simulations_data = agent.run_multiple_simulations(
            self.env, num_simulations, all_trials_data=self.all_trials_data, apply_microscope=apply_microscope, compute_likelihood=compute_likelihood)
        simulations_data['params'] = parameters
        return simulations_data
