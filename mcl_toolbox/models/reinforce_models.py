from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from mcl_toolbox.models.base_learner import Learner


class Policy(nn.Module):
    """Softmax Policy of the REINFORCE model

    Implemented in PyTorch so that automatic gradients can be computed.
    """

    def __init__(self, beta, num_features):
        super(Policy, self).__init__()
        self.num_features = num_features
        self.weighted_preference = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=num_features, bias=False
        )
        self.beta = Variable(torch.tensor(beta), requires_grad=False) #inverse temperature
        self.saved_log_probs = []
        self.term_log_probs = []
        self.rewards = []

    def forward(self, x, term_reward=None, termination=True):
        # x has size 13 x 1 x 51 with values between 0 and 1
        # y has size 13 x 1 x 1 with values -inf to + inf

        # add assertion check to see if x has nan values
        assert not torch.isnan(x).any(), f"feature matrix has nan values {x}"
        assert not torch.isinf(x).any(), f"feature matrix has inf values {x}"

        y = self.weighted_preference(x)
        if term_reward: #if TD, then append the estimated term_reward; else the value is learned like the other actions
            y[0][0] = torch.Tensor([term_reward])
        if not termination: #this one is independent of TD
            y[0][0] = torch.Tensor([-np.inf])
        action_scores = self.beta * y #beta should be high to be deterministic

        softmax_vals = F.log_softmax(action_scores, dim=0)
        # softmax_vals is tensor with length 13
        softmax_vals = torch.exp(softmax_vals)

        # softmax = e(softmax_vals) / sum(softmax_vals)
        # softmax_vals.sum() seems to be 1
        return softmax_vals / softmax_vals.sum()


class ValuePolicy(nn.Module):
    def __init__(self, num_features, num_actions):
        super(ValuePolicy, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.weighted_preference = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=num_features
        )
        self.linear = nn.Linear(num_actions, 1)
        self.baselines = []

    def init_weights(self):
        pass

    def forward(self, x):
        w_pref = self.weighted_preference(x).reshape(1, 1, -1)
        res = self.linear(w_pref)
        return res


class REINFORCE(Learner):
    """Base class of the REINFORCE model"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.lr = np.exp(params["lr"])
        self.gamma = np.exp(params["gamma"])
        self.beta = np.exp(params["inverse_temperature"])
        self.init_weights = np.array(params["priors"])
        # self.init_weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 100, 0, 5, 0, 0, 0])
        self.num_actions = attributes["num_actions"]
        self.no_term = attributes["no_term"]
        self.termination_value_known = attributes["termination_value_known"]
        self.policy = Policy(self.beta, self.num_features).double()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.init_model_params()
        self.action_log_probs = []
        self.term_rewards = []
        self.pseudo_rewards = []


    def init_model_params(self):
        # Initializing the parameters with people's priors.
        #todo:
        """
        /Users/rhe/PycharmProjects/mcl_toolbox/mcl_toolbox/models/reinforce_models.py:93: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:233.)
        [[self.init_weights * self.beta]]
        """
        self.policy.weighted_preference.weight.data = torch.DoubleTensor(
            [[self.init_weights * self.beta]]
        )

    def get_action_probs(self, env):
        available_actions = env.get_available_actions()
        X = np.zeros((self.num_actions, self.num_features))
        feature_state = env.get_feature_state()
        for action in available_actions:
            X[action] = feature_state[action]
        X = torch.DoubleTensor(X).view(self.num_actions, 1, self.policy.num_features)
        available_actions = torch.LongTensor(available_actions)
        X_new = X[available_actions] # 13 x 56; num_action x num_features
        if self.termination_value_known:
            term_reward = self.get_term_reward(env)
            probs = self.policy(X_new, term_reward, termination=not self.no_term)
            # IF term_reward is used, then the difference of prob between the term action and the other actions are much smaller
            # todo: why?
            # e.g. pid 24, strategy discovery with TD model 3325: 0.0723 for term, 0.077 for other actions
            # For same pid, 0.0008 for term and 0.0315 for action 1 using the PR model (3318)
            # term_reward is correctly implemented, e.g. in the beginning is it -5 because of env structure (check global vars).
            # after discoverin +1, term_reward becomes -4
        else:
            # calls the forward method in Policy class
            probs = self.policy(X_new, termination=not self.no_term)
        assert not torch.isnan(
            probs).any(), f"get_action_probs: feature matrix has nan values {probs}"
        assert not torch.isinf(
            probs).any(), f"get_action_probs: feature matrix has inf values {probs}"

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
        assert not torch.isnan(complete_probs).any(), f"get_action_details: feature matrix has nan values {complete_probs}"
        assert not torch.isinf(complete_probs).any(), f"get_action_details: feature matrix has inf values {complete_probs}"
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
        """
        Get the external rewards (returns) after end of episode.
        If learn from path, then only the external reward are taken here and the click cost are added in the if condition
        If not learn from path, then self.policy.rewards is a list consisting of click cost and the final reward after walking e.g. [-1, -1, ..., 200]
        Returns:

        """
        returns = []
        self.term_rewards.insert(0, 0)
        R = 0
        offset = 0
        if self.path_learn:
            # because every third reward is an external reward (because of third node)
            offset = 3
        # for i, r in enumerate(self.policy.rewards[:: -1 - offset]): # first and last one
        # I think it should be everything except the last 4 items and then reverse the order
        for i, r in enumerate(self.policy.rewards[:-(1+offset)][::-1]):
            pr = 0
            if self.use_pseudo_rewards:
                pr = self.pseudo_rewards[::-1][i]
            R = (r + pr) + self.gamma * R
            returns.insert(0, R) #includes both external final reward and the click costs [-1, -1, .. +50]
        if self.path_learn: # add the last 3 values from policy.reward
            # the last 3 values from policy.reward are the actual values underneath the nodes
            # for example [-1, -1, -1, -1, -1, 24, 0.0, 4, -4, 24], participant clicked 5 times,
            # received 24 (without click cost), the values underneath the path is 0 -> 4 -> -4 -> 24
            returns += self.policy.rewards[-3:]
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
        del self.policy.term_log_probs[:]
        del self.policy.saved_log_probs[:]
        self.pseudo_rewards = []
        return policy_loss.item()

    def get_current_weights(self):
        return torch.squeeze(self.policy.weighted_preference.weight.data).tolist() + [
            self.beta
        ]

    def learn_from_path(self, env, path):
        if self.path_learn:
            for node in path:
                reward = env.present_trial.node_map[node].value
                self.save_action_prob(env, node)
                self.policy.rewards.append(reward)
                env.step(node)

    def take_action(self, env, trial_info):
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
            # m is a tensor consisting of action probabilities (e.g. for small mouselab, tensor of len 13)
            m = self.get_action_details(env)
            action_tensor = torch.tensor(action)
            # likelihood of m (model action probabilities) and action_tensor (participant action)
            # e.g. m is a tensor of len 13, action tensor, selects the e.g. 5th action
            self.policy.saved_log_probs.append(m.log_prob(action_tensor))
            self.action_log_probs.append(m.log_prob(action_tensor).data.item())
        else:
            action = self.get_action(env)
        delay = env.get_feedback({"action": action})
        if self.compute_likelihood:
            s_next, r, done, _ = env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
            s_next, reward, done, taken_path = env.step(action)
        return action, reward, done, taken_path, delay

    def act_and_learn(self, env, trial_info={}):
        """

        Args:
            env:
            trial_info:

        Returns: action, reward, done, info

        """
        end_episode = False
        if "end_episode" in trial_info:
            end_episode = trial_info["end_episode"]
        policy_loss = 0
        if not end_episode:
            # get MER based on observed nodes
            term_reward = self.get_term_reward(env)
            self.term_rewards.append(term_reward)
            self.store_best_paths(env)
            # reward is the click cost
            action, reward, done, taken_path, delay = self.take_action(env, trial_info)
            # always appended but only used if pr is true
            # self.pseudo_reward is a list that is as long as number of clicks done because it reflects the PR for each clicking operation in sequence,
            # for example clicked 12, 5, 9, then PR is list of 3 reflecting how good 12, 5 and 9 was
            self.pseudo_rewards.append(self.get_pseudo_reward(env))
            self.policy.rewards.append(
                reward - self.subjective_cost - self.delay_scale * delay
            )
            if done:
                delay = env.get_feedback({"action": 0, "taken_path": taken_path})
                self.policy.rewards[-1] = reward - self.delay_scale * delay
                self.pseudo_rewards.append(self.get_pseudo_reward(env))
                if self.path_learn:
                    # updates policy.rewards with rewards of the take path
                    # this function is to attach the corresponding reward to be used in finish_episode
                    self.learn_from_path(env, taken_path)
                policy_loss = self.finish_episode()
            # if done = True, then reward = value of best_expected_path
            return action, reward, done, {"taken_path": taken_path, "loss": policy_loss}
        else:  # Should this model learn from the termination action when it's hierarchical?
            reward = 0
            taken_path = None
            if self.compute_likelihood:
                reward, taken_path, done = trial_info["participant"].make_click()
            if self.path_learn:
                self.learn_from_path(env, trial_info["taken_path"])
            # List of PR has to have same length as reward, todo: check if this is correct
            self.pseudo_rewards.append(self.get_pseudo_reward(env))
            self.finish_episode()
            return 0, reward, True, taken_path

    def simulate(self, env, compute_likelihood=False, participant=None):
        self.init_model_params()
        self.compute_likelihood = compute_likelihood
        self.action_log_probs = []
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        env.reset()
        policy_loss = 0
        for trial_num in range(num_trials):
            # print(f"Trial {trial_num}")
            trials_data["w"].append(self.get_current_weights())
            actions = []
            rewards = []
            self.previous_best_paths = []
            self.pseudo_rewards = []
            self.term_rewards = []
            done = False
            while not done:
                action, reward, done, info = self.act_and_learn(
                    env, trial_info={"participant": participant}
                )
                actions.append(action)
                rewards.append(reward)
                policy_loss += info["loss"]

                if done:
                    trials_data["taken_paths"].append(info["taken_path"])
            trials_data["r"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(rewards)
            env.get_next_trial()
        # add trial ground truths
        trials_data["envs"] = env.ground_truth
        if self.action_log_probs:
            trials_data["loss"] = -sum(self.action_log_probs)
        else:
            trials_data["loss"] = None
        return dict(trials_data)


class BaselineREINFORCE(REINFORCE):
    """Baseline version of the REINFORCE model"""

    def __init__(self, params, attributes):
        super().__init__(params, attributes)
        self.value_lr = np.exp(params["value_lr"])
        self.value_policy = ValuePolicy(self.num_features, self.num_actions).double()
        self.value_policy.weighted_preference.weight.data = torch.zeros_like(
            self.value_policy.weighted_preference.weight, requires_grad=True
        )
        self.value_optimizer = optim.Adam(
            self.value_policy.parameters(), lr=self.value_lr
        )

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
        # self.policy.term_log_probs.append(m.log_prob(0))
        self.value_policy.baselines.append(baseline)
        return action.item()

    def save_action_prob(self, env, action):
        m, baseline = self.get_action_details(env)
        action = torch.tensor(action)
        self.policy.saved_log_probs.append(m.log_prob(action))
        # self.policy.term_log_probs.append(m.log_prob(0))
        self.value_policy.baselines.append(baseline)

    def finish_episode(self):
        """Computing gradients and updating parameters."""
        policy_loss = []

        returns = self.get_end_episode_returns()
        returns = torch.tensor(returns).float()

        if self.value_policy.baselines:
            baselines = torch.tensor(
                self.value_policy.baselines[::-1], requires_grad=True
            ).float()
            value_loss = -(baselines.sum()).float()
            for log_prob, R, b in zip(self.policy.saved_log_probs, returns, baselines):
                policy_loss.append(-log_prob.float() * (R - b))
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
        # del self.policy.term_log_probs[:]
        return policy_loss.item()

    def take_action(self, env, trial_info):
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()
            action_tensor = torch.tensor(action)
            m, baseline = self.get_action_details(env)
            self.policy.saved_log_probs.append(m.log_prob(action_tensor))
            self.value_policy.baselines.append(baseline)
            self.action_log_probs.append(m.log_prob(action_tensor).data.item())
        else:
            action = self.get_action(env)
        delay = env.get_feedback({"action": action})
        if self.compute_likelihood:
            s_next, r, done, _ = env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
            s_next, reward, done, taken_path = env.step(action)
        return action, reward, done, taken_path, delay
