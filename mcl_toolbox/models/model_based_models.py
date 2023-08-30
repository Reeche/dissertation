import random
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import pyro
from pyro.distributions import DirichletMultinomial, Dirichlet
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.env.modified_mouselab import get_termination_mers
from hyperopt import STATUS_OK
from scipy.stats import norm, beta


class ModelBased(Learner):
    """Base class of the Model-based model"""

    def __init__(self, env, value_range, participant_obj, criterion, num_simulations, test_fitted_model):
        self.participant_obj = participant_obj
        self.env = env
        self.value_range = value_range
        self.action_log_probs = []
        self.pseudo_rewards = None
        self.num_available_nodes = len(self.env.get_available_actions()) - 1
        self.optimisation_criterion = criterion
        self.p_data = self.construct_p_data()
        self.num_simulations = num_simulations
        self.test_fitted_model = test_fitted_model

    def construct_p_data(self):
        p_data = {
            "envs": self.participant_obj.envs,
            "a": self.participant_obj.clicks,
            "mer": get_termination_mers(self.participant_obj.envs, self.participant_obj.clicks, self.env.pipeline),
            "r": self.participant_obj.rewards
        }
        return p_data

    def init_model_params(self, dist_alpha, dist_beta, alpha_multiplier):
        """
        Init the alpha values for the Dirichlet distribution. The alphas are randomly sampled from a Beta distribution
        with the parameters dist_alpha and dist_beta (those two are optimised). The random sample is then multiplied
        with a multiplier representing the prior belief of how often a value has been observed / how certain the pid is

        Args:
            dist_alpha: alpha of the beta distribution
            dist_beta: beta of the beta distribution
            alpha_multiplier: the higher, the more often a value has been observed

        Returns:

        """
        # each alpha parameter represents a possible value in the value_list, e.g. -48 to +48
        rv = beta(dist_alpha, dist_beta)
        x = np.linspace(beta.ppf(0.01, dist_alpha, dist_beta),
                        beta.ppf(0.99, dist_alpha, dist_beta), len(self.value_range))
        alpha = rv.pdf(x)
        alpha_list = alpha * alpha_multiplier
        dirichlet_alpha = dict(zip(self.value_range, alpha_list))
        self.dirichlet_alpha_dict = {}
        # alpha need to be n x m, e.g. 13 x range
        for i in range(1, self.num_available_nodes + 1):
            self.dirichlet_alpha_dict[i] = dirichlet_alpha.copy()
        return None

    def init_distributions(self):
        self.node_distributions = {}
        # create tensor of length value_range

        # create node distribution for all nodes
        for i in range(1, self.num_available_nodes + 1):
            self.node_distributions[i] = torch.distributions.Dirichlet(
                torch.tensor(list(self.dirichlet_alpha_dict[i].values())))
        return None

    def perform_updates(self, action):
        ## for the click that has been made, update the corresponding dirichlet alphas with the multinomial distribution (likelihood)
        # observed_value should only be the node value, i.e. model of the env and not model of the cost
        observed_value = self.env.ground_truth[self.env.present_trial_num][action]
        self.dirichlet_alpha_dict[action][int(observed_value)] += 1
        for i in range(1, self.num_available_nodes + 1):
            self.node_distributions[i] = DirichletMultinomial(
                concentration=torch.tensor((list(self.dirichlet_alpha_dict[i].values()))),
                total_count=self.env.present_trial_num + 1)  # because present_trial_number start at 0
            #todo: total_count = total_count=self.env.present_trial_num + 1 + round(sum(self.dirichlet_alpha_dict[i].values()))) ???
        return None

    @property
    def term_reward(self):
        """Get the max expected return in the current state"""
        pres_node_map = self.env.present_trial.node_map
        term_reward = pres_node_map[0].calculate_max_expected_return()
        #todo this is wrong, for pid 1, the term reward after first click should not be 0.0
        return term_reward

    def find_best_route(self, action, action_value=0):
        """
        Loops through all branches and get expected value of each branch /path
        Returns: the most/highest expected return

        """
        # find the best route
        expected_path_values = {}
        for i in list(self.env.present_trial.branch_map.keys()):
            path = self.env.present_trial.branch_map[i]
            expected_path_value = 0
            for node_num in path:
                node = self.env.present_trial.node_map[node_num]
                if node.observed:
                    expected_path_value += node.value
                else:
                    if node_num == 0:
                        expected_path_value = 0
                    elif node_num == action:
                        expected_path_value += action_value
            expected_path_values[i] = expected_path_value
        assert len(expected_path_values) == 6
        return max(expected_path_values.values())

    def mer_for_action(self, action):
        # given an action, get all possible values of the action and its probabilities
        # multiply the probabilities with MER
        mer = 0
        if action != 0:
            for index, value in enumerate(self.value_range):
                if isinstance(self.node_distributions[action], DirichletMultinomial):
                    # class concentration / total concentration * total count
                    prob = self.node_distributions[action].concentration[index] * \
                           self.node_distributions[action].total_count / \
                           sum(self.node_distributions[action].concentration)
                else:
                    sampled_probs = self.node_distributions[action].sample()
                    assert torch.round(sum(sampled_probs)) == 1, f"Sum is not 1 but {sum(sampled_probs)}"
                    prob = sampled_probs[index]
                    # prob = self.node_distributions[action].mean[index]
                assert prob <= 1, f"{prob} probability is larger than 1"
                mer += prob * self.find_best_route(action, value)
        elif action == 0:
            mer = self.find_best_route(action)
        return mer

    def node_depth(self, action):
        if action in [1, 5, 9]:
            return 1
        elif action in [2, 6, 10]:
            return 2
        elif action in [3, 4, 7, 8, 10, 11]:
            return 3

    def myopic_values(self) -> list:
        myopic_values = []
        for action in self.env.get_available_actions():  # all actions
            mer = self.mer_for_action(action)
            # print(mer, self.term_reward, self.env.cost(self.node_depth(action)))
            # myopic_value = mer - self.term_reward + self.env.cost(self.node_depth(action))
            myopic_value = mer - self.term_reward + self.env.cost(self.node_depth(action))
            myopic_values.append(myopic_value)
        return myopic_values

    def calculate_likelihood(self):
        myopic_values = self.myopic_values()
        softmax_vals = F.log_softmax(torch.tensor([x * self.inverse_temp for x in myopic_values]), dim=0)
        softmax_vals = torch.exp(softmax_vals)
        likelihood = softmax_vals / softmax_vals.sum()
        # for not available actions, insert 0
        for previous_actions in self.env.observed_action_list:
            likelihood = torch.cat((likelihood[:previous_actions], torch.tensor([0]), likelihood[previous_actions:]))
        return likelihood

    def sample_action(self) -> int:
        action_likelihood = self.calculate_likelihood()
        res = dict(zip(self.env.get_available_actions(), action_likelihood))  # mapping action to likelihood
        if np.sum(list(res.values())) != 0.0: #needed for upward compartibility with cluster python 3.10
            action = random.choices(list(res.keys()), weights=list(res.values()), k=1)
        else:
            action = random.choices(list(res.keys()), k=1)
        return action[0]

    def take_action(self, trial_info):
        if self.optimisation_criterion == "likelihood" and not self.test_fitted_model:
            pi = trial_info["participant"]
            action = pi.get_click()
            action_likelihood = self.calculate_likelihood()
            # how likely model would have taken the action
            self.action_log_probs.append(action_likelihood[action])
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
        self.inverse_temp = params['inverse_temp']
        if self.optimisation_criterion != "likelihood":
            self.sigma = params['sigma']

        simulations_data = defaultdict(list)
        for _ in range(self.num_simulations):
            trials_data = self.simulate(params['dist_alpha'], params['dist_beta'], params['alpha_multiplier'])
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
        return simulations_data

    def simulate(self, dist_alpha, dist_beta, alpha_multiplier):
        self.init_model_params(dist_alpha, dist_beta, alpha_multiplier)
        self.init_distributions()
        self.env.reset()
        # self.participant_obj.reset()  # resets number of trial and clicks, nothing else
        trials_data = defaultdict(list)
        num_trials = self.env.num_trials
        for trial_num in range(num_trials):
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
        trials_data["loss"] = self.objective_loss(trials_data)
        trials_data["status"] = STATUS_OK
        return dict(trials_data)

    def objective_loss(self, trials_data):
        # depending on the optimisation criterion, the loss the different
        if self.optimisation_criterion == "likelihood":
            loss = -np.sum(self.action_log_probs) #the likelihood should be maximised/the negative minimised, therefore the minus sign
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
