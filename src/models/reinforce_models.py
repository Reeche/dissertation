from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.base_learner import Learner
from src.utils.learning_utils import get_normalized_feature_values
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):
    """ Softmax Policy of the REINFORCE model

        Implemented in PyTorch so that automatic gradients can be computed.
    """

    def __init__(self, beta, num_features):
        super(Policy, self).__init__()
        self.num_features = num_features
        self.weighted_preference = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=num_features, bias=False)
        self.beta = Variable(torch.tensor(beta), requires_grad=False)
        self.saved_log_probs = []
        self.term_log_probs = []
        self.rewards = []

    def forward(self, x, term_reward=None, termination=True):
        x = self.weighted_preference(x)
        if term_reward:
            x[0][0] = torch.Tensor([term_reward])
        if not termination:
            x[0][0] = torch.Tensor([-np.inf])
        action_scores = self.beta*x
        softmax_vals = F.log_softmax(action_scores, dim=0)
        softmax_vals = torch.exp(softmax_vals)
        return softmax_vals/softmax_vals.sum()


class REINFORCE(Learner):
    """Base class of the REINFORCE model"""
    # TODO:
    # 2-stage REINFORCE
    def __init__(self, params, attributes):
        super().__init__()
        self.lr = np.exp(params['lr'])
        self.gamma = np.exp(params['gamma'])
        self.beta = np.exp(params['inverse_temperature'])
        self.num_actions = attributes['num_actions']
        self.init_weights = np.array(params['priors'])
        self.features = attributes['features']
        self.num_features = len(self.features)
        self.policy = Policy(self.beta, self.num_features).double()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.normalized_features = attributes['normalized_features']
        self.use_pseudo_rewards = attributes['use_pseudo_rewards']
        self.pr_weight = params['pr_weight']
        self.no_term = attributes['no_term']
        if 'delay_scale' in params:
            self.delay_scale = np.exp(params['delay_scale'])
        else:
            self.delay_scale = 0
        if 'subjective_cost' in params:
            self.subjective_cost = params['subjective_cost']
        else:
            self.subjective_cost = 0
        self.is_null = attributes['is_null']
        self.vicarious_learning = attributes['vicarious_learning']
        self.termination_value_known = attributes['termination_value_known']
        self.init_model_params()
        self.action_log_probs = []
        self.term_rewards = []

    def init_model_params(self):
        # Initializing the parameters with people's priors.
        self.policy.weighted_preference.weight.data = torch.DoubleTensor(
            [[self.init_weights*self.beta]])

    def get_action_probs(self, env):
        available_actions = env.get_available_actions()
        present_node_map = env.present_trial.node_map
        X = np.zeros((self.num_actions, self.num_features))
        for action in available_actions:
            X[action] = get_normalized_feature_values(
                present_node_map[action].compute_termination_feature_values(self.features), self.features, self.normalized_features)
        X = torch.DoubleTensor(X).view(self.num_actions, 1, self.policy.num_features)
        available_actions = torch.LongTensor(available_actions)
        X_new = X[available_actions]
        if self.termination_value_known:
            term_reward = self.get_term_reward(env)
            probs = self.policy(X_new, term_reward, termination=not self.no_term)
        else:
            probs = self.policy(X_new, termination=not self.no_term)
        complete_probs = torch.zeros(self.num_actions)
        for index, action in enumerate(available_actions):
            complete_probs[action] = probs[index]
        return complete_probs, X

    def get_action_details(self, env):
        """Generates action probabilities in the current state.

        Arguments:
            env {Gym env} -- Representation of the environment. 
        """
        complete_probs, _ = self.get_action_probs(env)
        m = Categorical(complete_probs)
        return m

    def get_action(self, env):
        m = self.get_action_details(env)
        action = m.sample()
        # Saving log-action probabilities to compute gradients at episode end.
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def save_action_prob(self, env, action):
        m = self.get_action_details(env)
        action = torch.tensor(action)
        self.policy.saved_log_probs.append(m.log_prob(action))

    def get_end_episode_returns(self):
        returns = []
        self.term_rewards.insert(0, 0)
        term_rewards = self.term_rewards[::-1]
        R = 0
        # In case of non-pr how is the term reward used?
        for i, r in enumerate(self.policy.rewards[::-1]):
            pr = 0
            if self.use_pseudo_rewards:
                pr = self.pr_weight*(term_rewards[i] - term_rewards[i+1])
            R = (r+pr) + self.gamma*R
            returns.insert(0, R)
        return returns

    def finish_episode(self):
        """
        Computing gradients and updating parameters.
        """
        policy_loss = []
        returns = self.get_end_episode_returns()
        returns = torch.tensor(returns)

        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        if policy_loss:
            policy_loss = torch.stack(policy_loss)
            policy_loss = policy_loss.sum()
        else:
            policy_loss = torch.tensor(0, dtype=torch.float, requires_grad=True)

        if not self.is_null:
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        del self.policy.term_log_probs[:]
        return policy_loss.item()

    def get_current_weights(self):
        return torch.squeeze(self.policy.weighted_preference.weight.data).tolist() + \
            [self.beta]

    def act_and_learn(self, env, end_episode=False):     
        if not end_episode:  
            action = self.get_action(env)
            term_reward = self.get_term_reward(env)
            self.term_rewards.append(term_reward)
            _, reward, done, info = env.step(action)
            self.policy.rewards.append(reward - self.subjective_cost)
            taken_path = info
            taken_action = action
            if done:
                self.finish_episode()
            return taken_action, reward, done, taken_path
        else:
            self.finish_episode()
            return None, None, None, None
        

    def simulate(self, env, compute_likelihood=False, participant=None):
        self.init_model_params()
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        env.reset()
        if compute_likelihood:
            first_trial_data = participant.first_trial_data
            all_trials_data = participant.all_trials_data
        policy_loss = 0
        loss = []
        for trial_num in range(num_trials):
            trials_data['w'].append(self.get_current_weights())
            self.term_rewards = []
            actions = []
            rewards = []
            if compute_likelihood:
                actions = all_trials_data['actions'][trial_num]
                rewards = all_trials_data['rewards'][trial_num]
                for action, reward in zip(actions, rewards):
                    self.save_action_prob(env, action)
                    term_reward = self.get_term_reward(env)
                    _, _, done, _ = env.step(action)
                    self.term_rewards.append(term_reward)
                    self.policy.rewards.append(reward - self.subjective_cost)
                    if done:
                        taken_path = first_trial_data['taken_path']
                        delay = env.present_trial.get_action_feedback(
                            taken_path)
                        self.policy.rewards[-1] = reward - self.delay_scale*delay
                        loss.append(torch.sum(torch.stack(self.policy.saved_log_probs)))
                        break
            else:
                self.num_actions = len(env.get_available_actions())
                while True:
                    action = self.get_action(env)
                    actions.append(action)
                    term_reward = self.get_term_reward(env)
                    self.term_rewards.append(term_reward)
                    s, reward, done, info = env.step(action)
                    self.policy.rewards.append(reward - self.subjective_cost)
                    rewards.append(reward)
                    if done:
                        taken_path = info
                        delay = env.present_trial.get_action_feedback(
                            taken_path)
                        self.policy.rewards[-1] = reward - \
                            self.delay_scale*delay
                        break
            trials_data['r'].append(np.sum(rewards))
            trials_data['rewards'].append(rewards)
            trials_data['a'].append(actions)
            env.get_next_trial()
            policy_loss += self.finish_episode()
        if loss:
            trials_data['loss'] = -torch.sum(torch.stack(loss)).data.cpu().numpy()
        else:
            trials_data['loss'] = None
        return dict(trials_data)

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

class BaselineREINFORCE(REINFORCE):
    """Baseline version of the REINFORCE model"""
    
    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.value_lr = np.exp(params['value_lr'])
        self.value_policy = ValuePolicy(self.num_features, self.num_actions).double()
        self.value_policy.weighted_preference.weight.data = torch.zeros_like(self.value_policy.weighted_preference.weight, requires_grad=True)
        self.value_optimizer = optim.Adam(self.value_policy.parameters(), lr = self.value_lr)

    def get_action_details(self, env):
        """Generates action probabilities in the current state.

        Arguments:
            env {Gym env} -- Representation of the environment. 
        """
        complete_probs, X = self.get_action_probs(env)
        m = Categorical(complete_probs)
        baseline = self.value_policy(X)
        return m, baseline

    def get_action(self, env):
        m, baseline = self.get_action_details(env)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        #self.policy.term_log_probs.append(m.log_prob(0))
        self.value_policy.baselines.append(baseline)
        return action.item()

    def save_action_prob(self, env, action):
        m, baseline = self.get_action_details(env)
        action = torch.tensor(action)
        self.policy.saved_log_probs.append(m.log_prob(action))
        #self.policy.term_log_probs.append(m.log_prob(0))
        self.value_policy.baselines.append(baseline)

    def finish_episode(self):
        """Computing gradients and updating parameters.
        """
        R = 0
        policy_loss = []
        value_loss = []

        returns = self.get_end_episode_returns()
        returns = torch.tensor(returns).float()

        if self.value_policy.baselines:
            baselines = torch.tensor(self.value_policy.baselines[::-1], requires_grad=True).float()
            value_loss = -(baselines.sum()).float()
            for log_prob, R, b in zip(self.policy.saved_log_probs, returns, baselines):
                policy_loss.append(-log_prob.float() * (R-b))
        else:
            value_loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        
        if policy_loss:
            policy_loss = torch.stack(policy_loss)
            policy_loss = policy_loss.sum().float()
        else:
            policy_loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.optimizer.step()
        self.value_optimizer.step()
        self.value_policy.baselines = []
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        #del self.policy.term_log_probs[:]
        return policy_loss.item()