import random
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from pyro.distributions import DirichletMultinomial
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.env.modified_mouselab import get_termination_mers
from hyperopt import STATUS_OK
from scipy.stats import norm, beta
from scipy.special import logsumexp
import math


class ModelBased(Learner):
    """Base class of the Model-based model"""

    def __init__(self, env, value_range, participant_obj, criterion, num_simulations, model, test_fitted_model):
        self.participant_obj = participant_obj
        self.env = env
        self.value_range = value_range
        self.action_log_probs = []
        self.pseudo_rewards = None
        self.num_available_nodes = 12
        self.optimisation_criterion = criterion
        self.p_data = self.construct_p_data()
        self.num_simulations = num_simulations
        self.test_fitted_model = test_fitted_model
        self.model = model

    def construct_p_data(self):
        p_data = {
            "envs": self.participant_obj.envs,
            "a": self.participant_obj.clicks,
            "mer": get_termination_mers(self.participant_obj.envs, self.participant_obj.clicks, self.env.pipeline),
            "r": self.participant_obj.rewards
        }
        return p_data

    def init_model_params(self, params):
        """
        Init the alpha values for the Dirichlet distribution. The alphas are randomly sampled from a Beta distribution
        with the parameters dist_alpha and dist_beta (those two are optimised)


        Args:
            dist_alpha: alpha of the beta distribution
            dist_beta: beta of the beta distribution
        Returns:

        """
        ## each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        if self.model == "uniform":
            rv = beta(np.exp(params["dist_alpha"]), np.exp(params["dist_beta"]))
            x = np.linspace(beta.ppf(0.01, np.exp(params["dist_alpha"]), np.exp(params["dist_beta"])),
                            beta.ppf(0.99, np.exp(params["dist_alpha"]), np.exp(params["dist_beta"])),
                            len(self.value_range))
            alpha = torch.tensor(rv.pdf(x))
            alpha_clipped = np.clip(alpha, 1e-8, 999999)
            dirichlet_alpha = dict(zip(self.value_range, alpha_clipped.tolist()))

            self.dirichlet_alpha_dict = {}
            # alpha need to be n x m, e.g. 13 x range
            for i in range(0, self.num_available_nodes + 1):
                self.dirichlet_alpha_dict[i] = dirichlet_alpha.copy()

        elif self.model == "linear":
            # depth of 1 for nodes 1, 5, 9; weight * 0 + intercept -> only intercept
            rv_1 = beta(np.exp(params["alpha_intercept"]), np.exp(params["beta_intercept"]))
            x_1 = np.linspace(beta.ppf(0.01, np.exp(params["alpha_intercept"]), np.exp(params["beta_intercept"])),
                              beta.ppf(0.99, np.exp(params["alpha_intercept"]), np.exp(params["beta_intercept"])),
                              len(self.value_range))
            alpha_level_1 = torch.tensor(rv_1.pdf(x_1))
            alpha_clipped_1 = np.clip(alpha_level_1, 1e-8, 999999)
            dirichlet_alpha_level_1 = dict(zip(self.value_range, alpha_clipped_1.tolist()))

            # depth of 2 for nodes 2, 6, 10; weight * 1 + intercept
            alpha_dist_2 = np.exp(params["alpha_weight"]) + np.exp(params["alpha_intercept"])
            beta_dist_2 = np.exp(params["beta_weight"]) + np.exp(params["beta_intercept"])
            rv_2 = beta(alpha_dist_2, beta_dist_2)
            x_2 = np.linspace(beta.ppf(0.01, alpha_dist_2, beta_dist_2), beta.ppf(0.99, alpha_dist_2, beta_dist_2),
                              len(self.value_range))
            alpha_level_2 = torch.tensor(rv_2.pdf(x_2))
            alpha_clipped_2 = np.clip(alpha_level_2, 1e-8, 999999)
            dirichlet_alpha_level_2 = dict(zip(self.value_range, alpha_clipped_2.tolist()))

            # depth for 3 for nodes 3, 4, 7, 8, 11, 12; weight * 2 + intercept
            alpha_dist_3 = 2 * np.exp(params["alpha_weight"]) + np.exp(params["alpha_intercept"])
            beta_dist_3 = 2 * np.exp(params["beta_weight"]) + np.exp(params["beta_intercept"])
            rv_3 = beta(alpha_dist_3, beta_dist_3)
            x_3 = np.linspace(beta.ppf(0.01, alpha_dist_3, beta_dist_3), beta.ppf(0.99, alpha_dist_3, beta_dist_3),
                              len(self.value_range))
            alpha_level_3 = torch.tensor(rv_3.pdf(x_3))
            alpha_clipped_3 = np.clip(alpha_level_3, 1e-8, 999999)
            dirichlet_alpha_level_3 = dict(zip(self.value_range, alpha_clipped_3.tolist()))

            self.dirichlet_alpha_dict = {0: dirichlet_alpha_level_1.copy(),
                                         1: dirichlet_alpha_level_1.copy(),
                                         2: dirichlet_alpha_level_2.copy(),
                                         3: dirichlet_alpha_level_3.copy(),
                                         4: dirichlet_alpha_level_3.copy(),
                                         5: dirichlet_alpha_level_1.copy(),
                                         6: dirichlet_alpha_level_2.copy(),
                                         7: dirichlet_alpha_level_3.copy(),
                                         8: dirichlet_alpha_level_3.copy(),
                                         9: dirichlet_alpha_level_1.copy(),
                                         10: dirichlet_alpha_level_2.copy(),
                                         11: dirichlet_alpha_level_3.copy(),
                                         12: dirichlet_alpha_level_3.copy()}

        elif self.model == "full":
            rv_1 = beta(np.exp(params["alpha_1"]), np.exp(params["beta_1"]))
            x_1 = np.linspace(beta.ppf(0.01, np.exp(params["alpha_1"]), np.exp(params["beta_1"])),
                              beta.ppf(0.99, np.exp(params["alpha_1"]), np.exp(params["beta_1"])),
                              len(self.value_range))
            alpha_1 = torch.tensor(rv_1.pdf(x_1))
            alpha_clipped_1 = np.clip(alpha_1, 1e-8, 999999)
            dirichlet_alpha_1 = dict(zip(self.value_range, alpha_clipped_1.tolist()))

            rv_2 = beta(np.exp(params["alpha_2"]) + np.exp(params["alpha_1"]),
                        np.exp(params["beta_2"]) + np.exp(params["beta_1"]))
            x_2 = np.linspace(beta.ppf(0.01, np.exp(params["alpha_2"]) + np.exp(params["alpha_1"]),
                                       np.exp(params["beta_2"]) + np.exp(params["beta_1"])),
                              beta.ppf(0.99, np.exp(params["alpha_2"]) + np.exp(params["alpha_1"]),
                                       np.exp(params["beta_2"]) + np.exp(params["beta_1"])),
                              len(self.value_range))
            alpha_2 = torch.tensor(rv_2.pdf(x_2))
            alpha_clipped_2 = np.clip(alpha_2, 1e-8, 999999)
            dirichlet_alpha_2 = dict(zip(self.value_range, alpha_clipped_2.tolist()))

            rv_3 = beta(np.exp(params["alpha_3"]) + np.exp(params["alpha_1"] + np.exp(params["alpha_2"]),
                                                           np.exp(params["beta_3"]) + np.exp(params["beta_1"]) + np.exp(
                                                               params["beta_2"])))
            x_3 = np.linspace(
                beta.ppf(0.01, np.exp(params["alpha_3"]) + np.exp(params["alpha_1"]) + np.exp(params["alpha_2"]),
                         np.exp(params["beta_3"]) + np.exp(params["beta_1"]) + np.exp(params["beta_2"])),
                beta.ppf(0.99, np.exp(params["alpha_3"]) + np.exp(params["alpha_1"]) + np.exp(params["alpha_2"]),
                         np.exp(params["beta_3"]) + np.exp(params["beta_1"]) + np.exp(params["beta_2"])),
                len(self.value_range))
            alpha_3 = torch.tensor(rv_3.pdf(x_3))
            alpha_clipped_3 = np.clip(alpha_3, 1e-8, 999999)
            dirichlet_alpha_3 = dict(zip(self.value_range, alpha_clipped_3.tolist()))

            self.dirichlet_alpha_dict = {0: dirichlet_alpha_1.copy(),
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
                                         12: dirichlet_alpha_3.copy()}

        elif self.model == "click_weight_only":
            rv_1 = beta(2, 1)
            x_1 = np.linspace(beta.ppf(0.01, 2, 1),
                              beta.ppf(0.99, 2, 1), len(self.value_range))
            alpha_1 = torch.tensor(rv_1.pdf(x_1))
            alpha_clipped_1 = np.clip(alpha_1, 1e-8, 999999)
            dirichlet_alpha_1 = dict(zip(self.value_range, alpha_clipped_1.tolist()))

            rv_2 = beta(1.5, 1)
            x_2 = np.linspace(beta.ppf(0.01, 1.5, 1),
                              beta.ppf(0.99, 1.5, 1), len(self.value_range))
            alpha_2 = torch.tensor(rv_2.pdf(x_2))
            alpha_clipped_2 = np.clip(alpha_2, 1e-8, 999999)
            dirichlet_alpha_2 = dict(zip(self.value_range, alpha_clipped_2.tolist()))

            rv_3 = beta(1.3, 1)
            x_3 = np.linspace(beta.ppf(0.01, 1.3, 1),
                              beta.ppf(0.99, 1.3, 1), len(self.value_range))
            alpha_3 = torch.tensor(rv_3.pdf(x_3))
            alpha_clipped_3 = np.clip(alpha_3, 1e-8, 999999)
            dirichlet_alpha_3 = dict(zip(self.value_range, alpha_clipped_3.tolist()))

            self.dirichlet_alpha_dict = {0: dirichlet_alpha_1.copy(),
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
                                         12: dirichlet_alpha_3.copy()}
        return None

    def init_distributions(self):
        self.node_distributions = {}
        # create node distribution for all nodes, including the starting node, whose distribution does not change
        for i in range(0, self.num_available_nodes + 1):
            self.node_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor(list(self.dirichlet_alpha_dict[i].values()), dtype=torch.float32))
        return None

    def perform_updates(self, action):
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        # observed_value should only be the node value, i.e. model of the env and not model of the cost
        observed_value = self.env.ground_truth[self.env.present_trial_num][action]
        self.dirichlet_alpha_dict[action][int(observed_value)] += 1 * self.click_weight
        # check of the node_distributions have been updated
        # old = hash(self.node_distributions)
        old = hash(frozenset(self.node_distributions.items()))
        for i in range(1, self.num_available_nodes):
            self.node_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor((list(self.dirichlet_alpha_dict[i].values()))),
                total_count=self.env.present_trial_num + 1)  # because present_trial_number start at 0
        new = hash(frozenset(self.node_distributions.items()))
        assert old != new, "Node distributions have not been updated"
        return None

    def node_depth(self, action):
        if action in [1, 5, 9]:
            return 1
        elif action in [2, 6, 10]:
            return 2
        elif action in [3, 4, 7, 8, 11, 12]:
            return 3
        elif action == 0:
            return 0

    def expectation_term(self, distribution):
        """
        Args:
            distribution: Dirichlet distribution of selected node
            value_range: range of possible values for the node

        Returns: expected value of the node given observations (alphas)

        """
        value_dict = {item: index for index, item in enumerate(self.value_range)}
        total_concentration = torch.sum(distribution.concentration)

        indices = np.array(list(value_dict.values()))
        probabilities = distribution.concentration[indices] / total_concentration
        values = torch.tensor(self.value_range, dtype=torch.float32)
        expectation = torch.sum(probabilities * values)

        return expectation

    def expectation_non_term(self, distribution, termination_value):
        value_dict = {item: index for index, item in enumerate(self.value_range)}
        total_concentration = torch.sum(distribution.concentration)

        indices = np.array(list(value_dict.values()))
        probabilities = distribution.concentration[indices] / total_concentration
        if self.model == "click_weight_only":
            expectation = np.sum([a * b for a, b in zip(probabilities, self.value_range)])
        else:
            max_termination_value = np.full(len(self.value_range), termination_value)
            values = np.maximum(self.value_range, max_termination_value)  # todo: check this??
            expectation = torch.sum(probabilities * values)
        return expectation

    def myopic_value(self, action):
        if action == 0:
            return self.env.get_term_reward()
        else:
            termination_value = self.expectation_term(self.node_distributions[0])
            return self.expectation_non_term(self.node_distributions[action], termination_value)

    def calculate_myopic_values(self) -> list:
        myopic_values_list = []
        # for action in self.env.get_available_actions():
        for action in range(0, self.num_available_nodes + 1):  # all actions
            if action in self.env.observed_action_list:
                myopic_vals = torch.tensor(self.env.cost(self.node_depth(action)))
                # myopic_value = torch.tensor(self.env.present_trial.ground_truth[action])
            else:
                myopic_vals = self.myopic_value(action) + self.env.cost(self.node_depth(action))
            myopic_values_list.append(myopic_vals)
        # print("myopic values", myopic_values_list)
        return myopic_values_list

    def calculate_likelihood(self):
        myopic_vals = self.calculate_myopic_values()
        myopic_vals = np.array([x * self.inverse_temp for x in myopic_vals])
        logsumexp_score = np.exp(myopic_vals - logsumexp(myopic_vals))
        action_probs = self.scale_probabilities(logsumexp_score)
        return action_probs

    def sample_action(self) -> int:
        # used when done fitting the parameter and simulate the model behaviour
        action_likelihood = self.calculate_likelihood()
        action_likelihood = self.scale_probabilities(action_likelihood)
        all_actions = list(range(0, self.num_available_nodes + 1))
        res = dict(zip(all_actions, action_likelihood))  # mapping action to likelihood

        # remove the already taken actions
        available_actions = self.env.get_available_actions()

        keys_to_remove = [key for key in res.keys() if key not in available_actions]
        for key in keys_to_remove:
            del res[key]

        action = random.choices(list(res.keys()), weights=list(res.values()), k=1)
        action = action[0]
        self.action_log_probs.append(np.log(action_likelihood[action]))

        return action

    def scale_probabilities(self, probabilities):
        # introduces bias towards some nodes, e.g. the immediate nodes
        total_sum = sum(probabilities)
        # bias_inner = 100
        for i in [1, 5, 9]:
            # probabilities[i] *= bias_inner
            # bias_inner *= 0.7 #bias decay
            probabilities[i] *= 1

        for j in [3, 4, 7, 8, 11, 12]:  # scale down outer nodes
            # probabilities[j] *= self.bias_outer
            probabilities[j] *= 1

        scaled_sum = sum(probabilities)
        scale_factor = total_sum / scaled_sum
        probabilities = [p * scale_factor for p in probabilities]
        return probabilities

    def take_action(self, trial_info):
        if self.optimisation_criterion == "likelihood" and not self.test_fitted_model:
            pi = trial_info["participant"]
            action = pi.get_click()

            # how likely model would have taken the action
            action_likelihood = self.calculate_likelihood()
            # if log underflow, that is probabily is too close to 0
            action_likelihood = np.clip(action_likelihood, 1e-8, 1)
            self.action_log_probs.append(np.log(action_likelihood[action]))

            reward, taken_path, done = pi.make_click()
            _, _, _, _ = self.env.step(action)  # needs to do this for env to mark node as observed
        else:
            action = self.sample_action()
            s_next, reward, done, taken_path = self.env.step(action)
        return action, reward, done, taken_path

    def act_and_learn(self, trial_info=None):
        if trial_info is None:
            trial_info = {}
        action, reward, done, taken_path = self.take_action(trial_info)
        if action != 0:
            self.perform_updates(action)
        return action, reward, done, taken_path

    def run_multiple_simulations(self, params):
        if self.model == "click_weight_only":
            self.inverse_temp = 1
        else:
            self.inverse_temp = params['inverse_temp']
        self.click_weight = params['click_weight']
        if self.optimisation_criterion != "likelihood":
            self.sigma = params['sigma']

        simulations_data = defaultdict(list)
        for _ in range(self.num_simulations):
            self.init_model_params(params)
            self.init_distributions()
            trials_data = self.simulate()
            for param in ["rewards", "a", "loss", "status"]:
                if param in trials_data:
                    simulations_data[param].append(trials_data[param])
            # reset participant, needed for likelihood object fxn
            self.participant_obj.reset()
        total_m_mers = []
        for i in range(len(simulations_data["a"])):
            m_mers = get_termination_mers(self.env.ground_truth, simulations_data["a"][i], self.env.pipeline)
            total_m_mers.append(m_mers)
        simulations_data["mer"] = total_m_mers
        simulations_data["loss"] = np.mean(simulations_data["loss"])
        simulations_data["status"] = simulations_data["status"][0]
        assert simulations_data["loss"] >= 0, f"Mean loss is not positive {simulations_data['loss']}"
        return simulations_data

    def simulate(self):
        self.env.reset()
        trials_data = defaultdict(list)
        num_trials = self.env.num_trials
        self.action_log_probs = []
        for trial_num in range(num_trials):
            # print("Trial", trial_num)
            actions, rewards = [], []
            done = False
            while not done:
                action, reward, done, taken_path = self.act_and_learn(
                    trial_info={"participant": self.participant_obj})
                rewards.append(reward)
                actions.append(action)
                if done:
                    trials_data["taken_paths"].append(taken_path)

            trials_data["rewards"].append(np.sum(rewards))
            trials_data["a"].append(actions)
            trials_data["costs"].append(rewards)
            self.env.get_next_trial()
        # add trial ground truths
        trials_data["envs"] = self.env.ground_truth
        trials_data["loss"] = self.calculate_loss(trials_data)
        trials_data["status"] = STATUS_OK
        return dict(trials_data)

    def calculate_loss(self, trials_data):
        if self.optimisation_criterion == "likelihood":
            # sum is negative, negation makes it positive
            loss = -np.sum(self.action_log_probs)
            # assert loss >= 0, f"loss is not positive {loss}"
        elif self.optimisation_criterion == "pseudo_likelihood":
            total_m_mers = []
            m_mers = get_termination_mers(self.env.ground_truth, trials_data["a"], self.env.pipeline)
            total_m_mers.append(m_mers)
            # not really calculating a mean because it is the result after 1 trial but used to flatten the list
            model_mer = np.mean(total_m_mers, axis=0)
            loss = -np.sum(
                [norm.logpdf(x, loc=y, scale=np.exp(self.sigma)) for x, y in zip(model_mer, self.p_data["mer"])])
            # if not logpdf, the values/signal are too small
        elif self.optimisation_criterion == "number_of_clicks_likelihood":
            # get the number of clicks of the participant and of the algorithm
            p_clicks = [len(sublist) - 1 for sublist in self.participant_obj.clicks]
            model_clicks = [len(sublist) - 1 for sublist in trials_data["a"]]
            loss = -np.sum([norm.logpdf(x, loc=y, scale=np.exp(self.sigma)) for x, y in zip(model_clicks, p_clicks)])
        else:
            UserWarning("Optimisation criterion not implemented")
        return loss
