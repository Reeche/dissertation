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
    """Base class of the Model-based model"""

    def __init__(self, env, value_range, term_range, participant_obj, criterion, num_simulations, node_assumption, update_rule,
                 compute_likelihood):
        self.participant_obj = participant_obj
        self.env = env
        self.value_range = value_range
        self.term_range = term_range
        self.action_log_probs = []
        self.pseudo_rewards = None
        self.num_available_nodes = 12
        self.optimisation_criterion = criterion
        self.p_data = self.construct_p_data()
        self.num_simulations = num_simulations
        self.update_rule = update_rule
        self.compute_likelihood = compute_likelihood
        self.node_assumption = node_assumption
        self.node_distributions = {}

    def construct_p_data(self):
        p_data = {
            # "envs": self.participant_obj.envs*2, #todo: why *2?
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

        "Uniform": for this model, all node values share the same set of parameters

        "Linear": for this model, the node values are linearly dependent of depth node.
        It is initialised with one alpha intercept + weight and one beta intercept + weight.
        alpha = alpha_intercept + alpha_weight * depth
        beta = beta_intercept + beta_weight * depth

        "Single": for this model, each node has its own set of alpha and beta

        "Level": for this model, all nodes on the same level share the same alpha and beta

        Args:
            dist_alpha: alpha of the beta distribution
            dist_beta: beta of the beta distribution
        Returns:

        """
        # todo: does it make sense that value of termination is the same as first level?
        ## each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        if self.node_assumption == "uniform":
            # all nodes share the same set of parameters
            rv = beta(np.exp(params["alpha"]), np.exp(params["beta"]))
            x = np.linspace(beta.ppf(0.01, np.exp(params["alpha"]), np.exp(params["beta"])),
                            beta.ppf(0.99, np.exp(params["alpha"]), np.exp(params["beta"])),
                            len(self.value_range))
            alpha = torch.tensor(rv.pdf(x))
            dirichlet_alpha = OrderedDict(zip(self.value_range, alpha.tolist()))

            self.dirichlet_alpha_dict = {}

            dirichlet_alpha_term = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))
            self.dirichlet_alpha_dict[0] = dirichlet_alpha_term.copy()

            # alpha need to be n x m, e.g. 13 x range
            for i in range(1, self.num_available_nodes + 1):
                self.dirichlet_alpha_dict[i] = dirichlet_alpha.copy()

        elif self.node_assumption == "level":
            # termination probability
            # rv_term = beta(params["alpha_term"], params["beta_term"])
            # x_term = np.linspace(beta.ppf(0.01, params["alpha_term"], params["beta_term"]),
            #                   beta.ppf(0.99, params["alpha_term"], params["beta_term"]),
            #                   len(self.term_range))
            # alpha_term = torch.tensor(rv_term.pdf(x_term))
            # dirichlet_alpha_term = OrderedDict(zip(self.term_range, alpha_term.tolist()))

            # each node has its own set of parameters
            rv_1 = beta(params["alpha_1"], params["beta_1"])
            x_1 = np.linspace(beta.ppf(0.01, params["alpha_1"], params["beta_1"]),
                              beta.ppf(0.99, params["alpha_1"], params["beta_1"]),
                              len(self.value_range))
            alpha_1 = torch.tensor(rv_1.pdf(x_1))
            dirichlet_alpha_1 = OrderedDict(zip(self.value_range, alpha_1.tolist()))

            rv_2 = beta(params["alpha_2"], params["beta_2"])
            x_2 = np.linspace(beta.ppf(0.01, params["alpha_2"], params["beta_2"]),
                              beta.ppf(0.99, params["alpha_2"], params["beta_2"]),
                              len(self.value_range))
            alpha_2 = torch.tensor(rv_2.pdf(x_2))
            dirichlet_alpha_2 = OrderedDict(zip(self.value_range, alpha_2.tolist()))

            rv_3 = beta(params["alpha_3"], params["beta_3"])
            x_3 = np.linspace(beta.ppf(0.01, params["alpha_3"], params["beta_3"]),
                              beta.ppf(0.99, params["alpha_3"], params["beta_3"]),
                              len(self.value_range))
            alpha_3 = torch.tensor(rv_3.pdf(x_3))
            dirichlet_alpha_3 = OrderedDict(zip(self.value_range, alpha_3.tolist()))

            dirichlet_alpha_term = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))

            self.dirichlet_alpha_dict = {0: dirichlet_alpha_term.copy(),
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

        elif self.node_assumption == "no_assumption":
            para_a = 1
            para_b = 1
            rv = beta(para_a, para_b)
            x = np.linspace(beta.ppf(0.01, para_a, para_b),
                            beta.ppf(0.99, para_a, para_b),
                            len(self.value_range))
            alpha = torch.tensor(rv.pdf(x))
            dirichlet_alpha = OrderedDict(zip(self.value_range, alpha.tolist()))

            # termination with 0 everything, i.e. init a dict with 1 everywhere
            dirichlet_alpha_term = OrderedDict(zip(self.term_range, [1] * len(self.term_range)))

            self.dirichlet_alpha_dict = {0: dirichlet_alpha_term.copy(),
                                         1: dirichlet_alpha.copy(),
                                         2: dirichlet_alpha.copy(),
                                         3: dirichlet_alpha.copy(),
                                         4: dirichlet_alpha.copy(),
                                         5: dirichlet_alpha.copy(),
                                         6: dirichlet_alpha.copy(),
                                         7: dirichlet_alpha.copy(),
                                         8: dirichlet_alpha.copy(),
                                         9: dirichlet_alpha.copy(),
                                         10: dirichlet_alpha.copy(),
                                         11: dirichlet_alpha.copy(),
                                         12: dirichlet_alpha.copy()}

        return None

    def init_distributions(self):
        for i in range(self.num_available_nodes + 1):
            alpha_values = list(self.dirichlet_alpha_dict[i].values())
            self.node_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor(alpha_values, dtype=torch.float32))
        return None

    def perform_updates(self, action):
        if self.update_rule == "level":
            print("Level update rule")
            observed_value = self.env.ground_truth[self.env.present_trial_num][action]
            if action in [1, 5, 9]:
                self.dirichlet_alpha_dict[1][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[5][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[9][int(observed_value)] += self.click_weight
            elif action in [2, 6, 10]:
                self.dirichlet_alpha_dict[2][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[6][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[10][int(observed_value)] += self.click_weight
            elif action in [3, 4, 7, 8, 11, 12]:
                self.dirichlet_alpha_dict[3][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[4][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[7][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[8][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[11][int(observed_value)] += self.click_weight
                self.dirichlet_alpha_dict[12][int(observed_value)] += self.click_weight

            for i in range(1, self.num_available_nodes + 1):
                alpha_values = list(self.dirichlet_alpha_dict[i].values())
                self.node_distributions[i] = DirichletMultinomial(
                    concentration=torch.tensor(alpha_values),
                    total_count=self.env.present_trial_num + 1
                )
            assert self.dirichlet_alpha_dict[1] == self.dirichlet_alpha_dict[5] == self.dirichlet_alpha_dict[
                9], "Distribution on the first level are not the same"
            assert self.dirichlet_alpha_dict[2] == self.dirichlet_alpha_dict[6] == self.dirichlet_alpha_dict[
                10], "Distribution on the second level are not the same"
            assert self.dirichlet_alpha_dict[3] == self.dirichlet_alpha_dict[4] == self.dirichlet_alpha_dict[7] == \
                   self.dirichlet_alpha_dict[8] == self.dirichlet_alpha_dict[11] == self.dirichlet_alpha_dict[
                       12], "Distribution on the third level are not the same"
        elif self.update_rule == "individual":
            print("Individual update rule")
            observed_value = self.env.ground_truth[self.env.present_trial_num][action]
            self.dirichlet_alpha_dict[action][int(observed_value)] += self.click_weight
            old_distributions = frozenset(self.node_distributions.items())
            for i in range(1, self.num_available_nodes + 1):
                alpha_values = list(self.dirichlet_alpha_dict[i].values())
                self.node_distributions[i] = DirichletMultinomial(
                    concentration=torch.tensor(alpha_values),
                    total_count=self.env.present_trial_num + 1
                )
            new_distributions = frozenset(self.node_distributions.items())
            assert old_distributions != new_distributions, "Node distributions have not been updated"
        else:
            raise ValueError("Update rule not recognised")
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

        # this is actually the same function as above but because we mignt need the commented lines below, so keep it here
        expectation = np.sum([a * b for a, b in zip(probabilities, self.value_range)])

        # We want the larger value between max_termination_value or the value_range
        # todo: something is wrong here, it makes the myopic values according to backward planning
        # max_termination_value = np.full(len(self.value_range), termination_value)
        # values = np.maximum(self.value_range, max_termination_value)
        # expectation = torch.sum(probabilities * values)
        return expectation

    def myopic_value(self, action):
        if action == 0:
            return self.env.get_term_reward()
        else:
            termination_value = self.expectation_term(self.node_distributions[0])
            return self.expectation_non_term(self.node_distributions[action], termination_value)

    def calculate_myopic_values(self) -> dict:
        # create empty dict with all actions as keys
        myopic_values_dict = OrderedDict.fromkeys(self.env.get_available_actions())
        for action in self.env.get_available_actions():
            myopic_vals = self.myopic_value(action) + self.env.cost(self.node_depth(action))
            myopic_values_dict[action] = myopic_vals
        # print("myopic values", myopic_values_dict)
        return myopic_values_dict

    def calculate_likelihood(self):
        myopic_vals_dict = self.calculate_myopic_values()
        myopic_vals = {key: value * self.inverse_temp for key, value in myopic_vals_dict.items()}

        logsum = logsumexp(list(myopic_vals.values()))
        action_probs = {key: np.exp(value - logsum) for key, value in myopic_vals.items()}

        return action_probs

    def sample_action(self) -> int:
        # used when done fitting the parameter and simulate the model behaviour
        action_likelihood = self.calculate_likelihood()
        # print("action likelihood", action_likelihood)
        action = random.choices(list(action_likelihood.keys()), weights=list(action_likelihood.values()), k=1)
        action = action[0]
        self.action_log_probs.append(np.log(action_likelihood[action]))

        return action

    def take_action(self, trial_info):  # Yash implementation
        if self.compute_likelihood:
            pi = trial_info["participant"]
            action = pi.get_click()

            # how likely model would have taken the action
            action_likelihood = self.calculate_likelihood()
            # if log underflow, that is probabily is too close to 0
            # action_likelihood = np.clip(action_likelihood, 1e-8, 1)
            self.action_log_probs.append(np.log(action_likelihood[action]))
        else:
            action = self.sample_action()
        if self.compute_likelihood:
            s_next, r, done, _ = self.env.step(action)
            reward, taken_path, done = pi.make_click()
        else:
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
