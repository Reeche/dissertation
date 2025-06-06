import random
from collections import defaultdict, OrderedDict
import torch
import numpy as np
from pyro.distributions import DirichletMultinomial
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.env.modified_mouselab import get_termination_mers
from hyperopt import STATUS_OK
from scipy.stats import norm, beta
from scipy.special import logsumexp


class ModelBased(Learner):
    def __init__(self, env, value_range, term_range, participant_obj, criterion, num_simulations, node_assumption, update_rule, compute_likelihood):
        self.env = env
        self.value_range = value_range
        self.term_range = term_range
        self.participant_obj = participant_obj
        self.optimisation_criterion = criterion
        self.num_simulations = num_simulations
        self.node_assumption = node_assumption
        self.update_rule = update_rule
        self.compute_likelihood = compute_likelihood
        self.num_available_nodes = 12
        self.p_data = self.construct_p_data()
        self.action_log_probs = []
        self.pseudo_rewards = None
        self.node_distributions = {}

    def construct_p_data(self):
        return {
            "envs": self.participant_obj.envs,
            "a": self.participant_obj.clicks,
            "mer": get_termination_mers(self.participant_obj.envs, self.participant_obj.clicks, self.env.pipeline),
            "r": self.participant_obj.rewards
        }

    def init_model_params(self, params):
        def generate_alpha_rv(alpha, beta_param):
            rv = beta(np.exp(alpha), np.exp(beta_param))
            x = np.linspace(beta.ppf(0.01, np.exp(alpha), np.exp(beta_param)),
                            beta.ppf(0.99, np.exp(alpha), np.exp(beta_param)),
                            len(self.value_range))
            return OrderedDict(zip(self.value_range, torch.tensor(rv.pdf(x)).tolist()))

        if self.node_assumption == "uniform":
            alpha_dict = generate_alpha_rv(params["alpha"], params["beta"])
            term_dict = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))
            self.dirichlet_alpha_dict = {0: term_dict.copy()}
            for i in range(1, self.num_available_nodes + 1):
                self.dirichlet_alpha_dict[i] = alpha_dict.copy()

        elif self.node_assumption == "level":
            dirichlet_alpha_1 = generate_alpha_rv(params["alpha_1"], params["beta_1"])
            dirichlet_alpha_2 = generate_alpha_rv(params["alpha_2"], params["beta_2"])
            dirichlet_alpha_3 = generate_alpha_rv(params["alpha_3"], params["beta_3"])
            term_dict = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))

            self.dirichlet_alpha_dict = {
                0: term_dict.copy(),
                1: dirichlet_alpha_1.copy(),
                2: dirichlet_alpha_2.copy(),
                3: dirichlet_alpha_3.copy(),
                4: dirichlet_alpha_3.copy(),
                5: dirichlet_alpha_1.copy(),
                6: dirichlet_alpha_2.copy(),
                7: dirichlet_alpha_3.copy(),
                8: dirichlet_alpha_3.copy(),
                9: dirichlet_alpha_1.copy(),
                10: dirichlet_alpha_2.copy(),
                11: dirichlet_alpha_3.copy(),
                12: dirichlet_alpha_3.copy()
            }

        elif self.node_assumption == "no_assumption":
            alpha_dict = generate_alpha_rv(1, 1)
            term_dict = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))
            self.dirichlet_alpha_dict = {0: term_dict.copy()}
            for i in range(1, self.num_available_nodes + 1):
                self.dirichlet_alpha_dict[i] = alpha_dict.copy()

    def init_distributions(self):
        for i in range(self.num_available_nodes + 1):
            alpha_values = list(self.dirichlet_alpha_dict[i].values())
            self.node_distributions[i] = torch.distributions.Dirichlet(torch.tensor(alpha_values, dtype=torch.float32))

    def perform_updates(self, action):
        observed_value = self.env.ground_truth[self.env.present_trial_num][action]
        if self.update_rule == "level":
            level_map = {
                (1, 5, 9): [1, 5, 9],
                (2, 6, 10): [2, 6, 10],
                (3, 4, 7, 8, 11, 12): [3, 4, 7, 8, 11, 12]
            }
            for keys, nodes in level_map.items():
                if action in keys:
                    for node in nodes:
                        self.dirichlet_alpha_dict[node][int(observed_value)] += self.click_weight
            for i in range(1, self.num_available_nodes + 1):
                alpha_values = list(self.dirichlet_alpha_dict[i].values())
                self.node_distributions[i] = DirichletMultinomial(torch.tensor(alpha_values), total_count=self.env.present_trial_num + 1)
        elif self.update_rule == "individual":
            self.dirichlet_alpha_dict[action][int(observed_value)] += self.click_weight
            for i in range(1, self.num_available_nodes + 1):
                alpha_values = list(self.dirichlet_alpha_dict[i].values())
                self.node_distributions[i] = DirichletMultinomial(torch.tensor(alpha_values), total_count=self.env.present_trial_num + 1)
        else:
            raise ValueError("Update rule not recognised")

    def node_depth(self, action):
        return {0: 0, 1: 1, 5: 1, 9: 1, 2: 2, 6: 2, 10: 2, 3: 3, 4: 3, 7: 3, 8: 3, 11: 3, 12: 3}.get(action, -1)

    def expectation_term(self, distribution):
        value_dict = {item: index for index, item in enumerate(self.value_range)}
        indices = np.array(list(value_dict.values()))
        probs = distribution.concentration[indices] / torch.sum(distribution.concentration)
        values = torch.tensor(self.value_range, dtype=torch.float32)
        return torch.sum(probs * values)

    def expectation_non_term(self, distribution, termination_value):
        value_dict = {item: index for index, item in enumerate(self.value_range)}
        indices = np.array(list(value_dict.values()))
        probs = distribution.concentration[indices] / torch.sum(distribution.concentration)
        return np.sum([a * b for a, b in zip(probs, self.value_range)])

    def myopic_value(self, action):
        if action == 0:
            return self.env.get_term_reward()
        termination_value = self.expectation_term(self.node_distributions[0])
        return self.expectation_non_term(self.node_distributions[action], termination_value)

    def calculate_myopic_values(self):
        return OrderedDict((action, self.myopic_value(action) + self.env.cost(self.node_depth(action)))
                           for action in self.env.get_available_actions())

    def calculate_likelihood(self):
        myopic_vals = {k: v * self.inverse_temp for k, v in self.calculate_myopic_values().items()}
        logsum = logsumexp(list(myopic_vals.values()))
        return {k: np.exp(v - logsum) for k, v in myopic_vals.items()}

    def sample_action(self):
        action_likelihood = self.calculate_likelihood()
        action = random.choices(list(action_likelihood.keys()), weights=list(action_likelihood.values()), k=1)[0]
        self.action_log_probs.append(np.log(action_likelihood[action]))
        return action

    def take_action(self, trial_info):
        pi = trial_info.get("participant") if trial_info else None
        if self.compute_likelihood:
            action = pi.get_click()
            action_likelihood = self.calculate_likelihood()
            self.action_log_probs.append(np.log(action_likelihood[action]))
        else:
            action = self.sample_action()
        s_next, reward, done, taken_path = self.env.step(action)
        if self.compute_likelihood:
            reward, taken_path, done = pi.make_click()
        return action, reward, done, taken_path

    def act_and_learn(self, trial_info=None):
        action, reward, done, taken_path = self.take_action(trial_info or {})
        if action != 0:
            self.perform_updates(action)
        return action, reward, done, taken_path

    def run_multiple_simulations(self, params):
        self.inverse_temp = params['inverse_temp']
        self.click_weight = params['click_weight']
        if self.optimisation_criterion != "likelihood":
            self.sigma = params['sigma']

        simulations_data = defaultdict(list)
        for _ in range(self.num_simulations):
            self.init_model_params(params)
            self.init_distributions()
            trials_data = self.simulate()
            for key in ["rewards", "a", "loss", "status"]:
                if key in trials_data:
                    simulations_data[key].append(trials_data[key])
            self.participant_obj.reset()

        simulations_data["mer"] = [get_termination_mers(self.env.ground_truth, a, self.env.pipeline)
                                   for a in simulations_data["a"]]
        simulations_data["loss"] = np.mean(simulations_data["loss"])
        simulations_data["status"] = simulations_data["status"][0]
        return simulations_data

    def simulate(self):
        self.env.reset()
        trials_data = defaultdict(list)
        self.action_log_probs = []

        for _ in range(self.env.num_trials):
            actions, rewards, done = [], [], False
            while not done:
                action, reward, done, taken_path = self.act_and_learn({"participant": self.participant_obj})
                actions.append(action)
                rewards.append(reward)
                if done:
                    trials_data["taken_paths"].append(taken_path)
            trials_data["rewards"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(rewards)
            self.env.get_next_trial()

        trials_data["envs"] = self.env.ground_truth
        trials_data["loss"] = self.calculate_loss(trials_data)
        trials_data["status"] = STATUS_OK
        return dict(trials_data)

    def calculate_loss(self, trials_data):
        if self.optimisation_criterion == "likelihood":
            return -np.sum(self.action_log_probs)
        elif self.optimisation_criterion == "pseudo_likelihood":
            model_mer = np.mean([get_termination_mers(self.env.ground_truth, trials_data["a"], self.env.pipeline)], axis=0)
            return -np.sum([norm.logpdf(x, loc=y, scale=np.exp(self.sigma)) for x, y in zip(model_mer, self.p_data["mer"])])
        elif self.optimisation_criterion == "number_of_clicks_likelihood":
            p_clicks = [len(sublist) - 1 for sublist in self.participant_obj.clicks]
            model_clicks = [len(sublist) - 1 for sublist in trials_data["a"]]
            return -np.sum([norm.logpdf(x, loc=y, scale=np.exp(self.sigma)) for x, y in zip(model_clicks, p_clicks)])
        else:
            return float('inf')
