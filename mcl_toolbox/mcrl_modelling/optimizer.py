import json
import os
import logging
from pathlib import Path

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
import seaborn as sns
from hyperopt import Trials, fmin, hp, tpe
from pyabc.transition import MultivariateNormalTransition

from mcl_toolbox.env.modified_mouselab import get_termination_mers
from mcl_toolbox.models.hierarchical_models import HierarchicalLearner
from mcl_toolbox.models.lvoc_models import LVOC
from mcl_toolbox.models.reinforce_models import REINFORCE, BaselineREINFORCE
from mcl_toolbox.models.rssl_models import RSSL
from mcl_toolbox.models.sdss_models import SDSS
from mcl_toolbox.utils.learning_utils import (compute_objective,
                                              get_relevant_data)
from mcl_toolbox.utils.participant_utils import ParticipantIterator

loggers_to_shut_up = [
    "hyperopt.tpe",
    "hyperopt.fmin",
    "hyperopt.pyll.base",
]
for logger in loggers_to_shut_up:
    logging.getLogger(logger).setLevel(logging.ERROR)

models = {
    "lvoc": LVOC,
    "rssl": RSSL,
    "hierarchical_learner": HierarchicalLearner,
    "sdss": SDSS,
    "reinforce": REINFORCE,
    "baseline_reinforce": BaselineREINFORCE,
}

mcrl_modelling_dir = Path(__file__).parents[0]
model_dir = Path(__file__).parents[1].joinpath("models")

param_config = json.load(
    open(mcrl_modelling_dir.joinpath("param_search_space.json"), "rb")
)
model_config = json.load(open(mcrl_modelling_dir.joinpath("model_params.json"), "rb"))
model_details = json.load(open(model_dir.joinpath("models.json"), "rb"))


def hyperopt_space(params_list):
    """Should return a dict of the form required by hyperopt

    Arguments:
        params {[list]} -- List of param configs
    """
    space = {}
    for param, param_type, param_range in params_list:
        if param_type != "constant":
            a = param_range[0]
            b = param_range[1]
            if param_type == "uniform":
                space[param] = hp.uniform(param, a, b)
            elif param_type == "loguniform":
                # space[param] = hp.loguniform(param, np.log(a), np.log(b)) # Verify this
                # This change is to maintain uniformity
                space[param] = hp.uniform(param, np.log(a), np.log(b))
            elif param_type == "quniform":
                space[param] = hp.quniform(param, a, b, 1)
            elif param_type == "normal":
                space[param] = hp.normal(param, a, b)
        else:
            space[param] = param_range
    space["lik_sigma"] = hp.uniform("lik_sigma", np.log(1e-3), np.log(1e3))
    return space


def pyabc_prior(params_list):
    """Should return a dict of the form required by pyabc

    Arguments:
        params {[list]} -- List of param configs
    """
    prior = {}
    for param, param_type, param_range in params_list:
        if param_type != "constant":
            a = param_range[0]
            b = param_range[1]
            if param_type == "uniform" or param_type == "quniform":
                # pyabc does not yet support discrete uniform well
                prior[param] = pyabc.RV("uniform", a, b - a)
            elif param_type == "loguniform":
                # prior[param] = pyabc.RV("loguniform", *param_range)
                log_param_range = np.log(param_range)
                prior[param] = pyabc.RV(
                    "uniform",
                    log_param_range[0],
                    log_param_range[1] - log_param_range[0],
                )
            elif param_type == "normal":
                prior[param] = pyabc.RV("norm", *param_range)
        else:
            prior[param] = pyabc.RV("uniform", param_range, 0)
    prior = pyabc.Distribution(**prior)
    return prior


def param_info(param_dict, key):
    return (key, param_dict["type"], param_dict["range"])


def get_params_list(params_dict):
    params_list = []
    for param in params_dict:
        if params_dict[param]:
            params_list.append(param_info(params_dict[param], param))
    return params_list


def get_params(params, param_config):
    params_list = []
    for param in params:
        params_list.append(param_info(param_config["model_params"][param], param))
    return params_list


def make_constant(constant_value):
    return {"type": "constant", "range": constant_value}


def make_prior(param_dict, num_priors, bandit_prior=False):
    params_list = []
    t = "prior"
    if bandit_prior:
        t = "bandit_prior"
    for i in range(num_priors):
        params_list.append(param_info(param_dict, f"{t}_{i}"))
    return params_list


def parse_config(
    learner, learner_attributes, hierarchical=False, hybrid=False, general_params=False
):
    params_list = []
    bandit_prior = False
    learner_params = model_config[learner]
    param_models = param_config["model_params"]

    # Add base params
    params_list += get_params(learner_params["params"], param_config)

    # Adding extra params if they are in attributes and have a value True
    extra_params = learner_params["extra_params"]
    for i, param in enumerate(extra_params):
        if param in learner_attributes and learner_attributes[param]:
            params_list.append(param_info(param_models[param], param))
        # else:
        #     con = learner_params["extra_param_defaults"][i]
        #     params_list.append(param_info(make_constant(con), param))

    # General params
    if general_params:
        if "pr_weight" in learner_attributes:
            params_list.append(param_info(param_models["pr_weight"], "pr_weight"))
        else:
            params_list.append(param_info(make_constant(1), "pr_weight"))

    if hierarchical:
        decision_rule = learner_attributes["decision_rule"]
        actor = learner_attributes["actor"]
        params_list += get_params_list(param_config["decision_params"][decision_rule])
        params_list += parse_config(actor, learner_attributes, False, False, False)
    elif hybrid:
        selector = learner_attributes["selector"]
        learner = learner_attributes["learner"]
        params_list += parse_config(selector, learner_attributes, False, False, False)
        params_list += parse_config(learner, learner_attributes, False, False, False)
    else:
        if "prior" in learner_attributes:
            prior = learner_attributes["prior"]
            num_priors = learner_attributes["num_priors"]
            param_dict = param_models[prior]
            params_list += make_prior(param_dict, num_priors, False)
            if prior == "gaussian_prior":
                param = "gaussian_var"
                params_list.append(
                    param_info(param_config["model_params"][param], param)
                )
    return params_list


def get_space(learner, learner_attributes, optimizer="pyabc"):
    hierarchical = False
    hybrid = False
    if learner == "hierarchical_learner":
        hierarchical = True
    if learner == "sdss":
        hybrid = True
    params_list = parse_config(learner, learner_attributes, hierarchical, hybrid, True)
    if optimizer == "pyabc":
        return pyabc_prior(params_list)
    else:
        return hyperopt_space(params_list)


def construct_p_data(participant, pipeline):
    p_data = {
        "envs": participant.envs,
        "a": participant.clicks,
        "s": participant.strategies,
        "mer": get_termination_mers(participant.envs, participant.clicks, pipeline),
        "r": participant.scores,
        "w": participant.weights if "weights" in dir(participant) else None,
    }
    return p_data


def construct_objective_fn(optimizer, objective, p_data, pipeline):
    # construct objective function based on the selected optimizer and objective
    objective_fn = lambda x, y: compute_objective(objective, x, p_data, pipeline)
    if optimizer == "pyabc":
        # todo: currently, the pyabc is not able to optimize these three criteria
        if objective in ["reward", "strategy_accuracy", "clicks_overlap"]:
            objective_fn = lambda x, y: -compute_objective(objective, x, y, pipeline)
        else:
            objective_fn = lambda x, y: compute_objective(objective, x, y, pipeline)
    return objective_fn


def optimize_hyperopt_params(
    objective_fn,
    param_ranges,
    max_evals=100,
    trials=True,
    method=tpe.suggest,
    init_evals=30,
    show_progressbar=False,
    rstate=None
):
    estimator = partial(method, n_startup_jobs=init_evals)
    trials = Trials() if trials else None
    best_params = fmin(
        fn=objective_fn,
        space=param_ranges,
        algo=estimator,
        max_evals=max_evals,
        trials=trials,
        show_progressbar=show_progressbar,
        rstate=rstate,
    )
    return best_params, trials


def estimate_pyabc_posterior(
    model, prior, distance_fn, observation, db_path, eps=0.1, num_populations=10
):
    """
    See if this can be made to use the model selection function
    """
    transition = MultivariateNormalTransition(scaling=0.1)
    abc = pyabc.ABCSMC(
        model, prior, distance_fn, transitions=[transition], population_size=20
    )  # Change this
    abc.new(db_path, observation)
    history = abc.run(minimum_epsilon=eps, max_nr_populations=num_populations)
    return history


def combine_priors(params, num_priors, prefix="prior"):
    init_weights = np.zeros(num_priors)
    for i in range(num_priors):
        init_weights[i] = params[f"{prefix}_{i}"]
    return init_weights


class ParameterOptimizer:
    def __init__(self, learner, learner_attributes, participant, env):
        self.learner = learner
        self.learner_attributes = learner_attributes
        self.participant = participant
        self.env = env
        self.pipeline = self.env.pipeline
        self.compute_likelihood = False
        if self.learner in ["sdss"]:
            self.model = models[self.learner_attributes["learner"]]
        elif self.learner in ["hierarchical_learner"]:
            self.model = models[self.learner_attributes["actor"]]
        self.reward_data = []
        self.click_data = []
        self.agent = None

    def objective_fn(self, params, get_sim_data=False):
        """
        This function takes the selected parameters, created an agent with those parameters and run simulations

        Args:
            params: parameters
            get_sim_data:

        Returns: relevant data according to the learner

        """
        # returns relevant_data, which contains e.g. reward, loss
        features = self.learner_attributes['features']
        num_priors = self.learner_attributes['num_priors']
        priors = combine_priors(params, num_priors)
        params['priors'] = priors
        if self.learner == "sdss":
            num_strategies = int(params['num_strategies'])
            bandit_params = np.ones(2 * num_strategies)
            bandit_params[:num_strategies] *= params['alpha']
            bandit_params[num_strategies:] *= params['beta']
            params['bandit_params'] = bandit_params
            self.learner_attributes['learner'] = self.model
            self.learner_attributes['strategy_space'] = list(range(num_strategies))
        elif self.learner == "hierarchical_learner":
            self.learner_attributes['actor'] = self.model

        # the agent is the selected model with corresponding priors to be fitted
        self.agent = models[self.learner](params, self.learner_attributes)
        del params['priors']
        if self.learner == "sdss":
            del params["bandit_params"]

        print("objective function")
        print(self.compute_likelihood)
        simulations_data = self.agent.run_multiple_simulations(
            self.env,
            self.num_simulations,
            participant=ParticipantIterator(self.participant, click_cost=-self.env.cost(0)),
            compute_likelihood=self.compute_likelihood,
        )
        relevant_data = get_relevant_data(simulations_data, self.objective)
        if self.objective in [
            "mer_performance_error",
            "pseudo_likelihood",
            "reward",
            "likelihood",
        ]:
            self.reward_data.append(relevant_data["mer"])
        if self.objective == "pseudo_likelihood":
            relevant_data['sigma'] = params['lik_sigma']
        if self.objective == "clicks_overlap":
            self.click_data.append(relevant_data["a"])
            self.reward_data.append(relevant_data["mer"])
        if self.objective == "number_of_clicks_likelihood":
            self.click_data.append(relevant_data["a"])
            self.reward_data.append(relevant_data["mer"])
            relevant_data['sigma'] = params['lik_sigma']
        if get_sim_data:
            return relevant_data, simulations_data#, self.agent
        else:
            return relevant_data

    def get_prior(self):
        return get_space(self.learner, self.learner_attributes, self.optimizer)

    def optimize(self, objective, num_simulations=1, optimizer="pyabc",
                 db_path="sqlite:///test.db", compute_likelihood=False,
                 max_evals=100, rstate=None):
        """
        This function first gets the relevant participant data,
        creates a lambda function as required by fmin function
        Calling the lambda function creates simulated data depending on num_simulation
        The lambda function is called max_evals times in optimize_hyperopt_params.

        Example: num_simulation: 30, max_evals: 400, model: reinforce
        The model is initated with a set of parameters and creates simulated data for 30 runs
        The data for the 30 runs is passed on to the optimizer (optimize_hyperopt_params -> fmin) and parameters are
        optimised based on the 30 runs and participant data.
        Then the updated parameters are passed to the model and another 30 runs are created with the new parameters
        The loop continues 400 times.

        Args:
            objective: str e.g. "likelihood" or "pseudo-likelihood"
            num_simulations: integer
            optimizer: str, e.g. "hyperopt"
            db_path: path to database
            compute_likelihood: boolean
            max_evals: integer
            rstate: random state for hyperopt

        Returns: res: results

        """
        self.objective = objective
        self.compute_likelihood = compute_likelihood
        self.num_simulations = num_simulations
        self.optimizer = optimizer
        prior = self.get_prior()
        # get participant data as dict
        p_data = construct_p_data(self.participant, self.pipeline)
        self.p_data = p_data
        # returns a loss but is only "triggered" later in line 404
        distance_fn = construct_objective_fn(optimizer, objective, p_data, self.pipeline)
        # filter participant data by only the relevant data depending on objective
        observation = get_relevant_data(p_data, self.objective)
        if objective == "likelihood":
            self.compute_likelihood = True
        if optimizer == "pyabc":
            res = estimate_pyabc_posterior(self.objective_fn, prior, distance_fn, observation,
                                           db_path, num_populations=5)
        else:
            lambda_objective_fn = lambda x: distance_fn(self.objective_fn(x), p_data)
            res = optimize_hyperopt_params(lambda_objective_fn, prior, max_evals=max_evals,
                                           show_progressbar=True, rstate=rstate)  # returns best parameters (res) and trials
        return res, prior, self.objective_fn

    def fit_with_params(self, objective, params, compute_likelihood=False):

        self.objective = objective
        self.compute_likelihood = compute_likelihood

        # get participant data as dict
        p_data = construct_p_data(self.participant, self.pipeline)
        self.p_data = p_data

        # filter participant data by only the relevant data depending on objective
        observation = get_relevant_data(p_data, self.objective)
        if objective == "likelihood":
            self.compute_likelihood = True


        return self.objective_fn(params, get_sim_data=True)

    def run_model(self, params, objective, num_simulations=1, optimizer="pyabc",
                  db_path="sqlite:///test.db"):
        self.objective = objective
        self.num_simulations = num_simulations
        p_data = construct_p_data(self.participant, self.pipeline)
        data = self.objective_fn(params)
        return data, p_data

    def run_hp_model(self, params, objective, num_simulations=1):
        self.objective = objective
        self.num_simulations = num_simulations
        p_data = construct_p_data(self.participant, self.pipeline)
        print("Running simulations: ")
        print(self.compute_likelihood)
        data = self.objective_fn(params, get_sim_data=True)
        return data, p_data

    def run_hp_model_nop(self, params, objective, num_simulations=1):
        self.objective = objective
        self.num_simulations = num_simulations
        # p_data = construct_p_data(self.participant, self.pipeline)
        p_data = {"mer": []}
        data = self.objective_fn(params, get_sim_data=True)
        return data, p_data

    def plot_rewards(self, i=0, path="", plot=True):
        data = []
        for j in range(len(self.reward_data[i])):
            for k in range(len(self.reward_data[i][j])):
                data.append([k + 1, self.reward_data[i][j][k], "algo"])  # reward data of the algorithm
        p_mer = self.p_data["mer"]  # reward data of the participant
        for i, m in enumerate(p_mer):
            data.append([i + 1, m, "participant"])
        reward_data = pd.DataFrame(data, columns=["Number of trials", "Reward", "Type"])
        if plot:
            ax = sns.lineplot(x="Number of trials", y="Reward", hue="Type", data=reward_data)
            plt.savefig(path, bbox_inches='tight')
            #plt.show()
            plt.close()
        return reward_data

    def plot_clicks(self, i=0, path="", plot=True):

        # # number of clicks of the algorithm
        # data = []
        # for j in range(len(self.click_data[i])):
        #     for k in range(len(self.click_data[i][j])):
        #         data.append([k + 1, len(self.click_data[i][j][k]), "algo"])  # reward data of the algorithm

        algo_num_actions = dict((k, []) for k in range(len(self.click_data[0])))
        for j in range(len(self.click_data[i])):
            for k in range(len(self.click_data[i][j])):
                algo_num_actions[k] = len(self.click_data[i][j][k])

        # number of clicks of the participants
        p_actions = self.p_data["a"]  # reward data of the participant
        p_num_actions = dict((k, []) for k in range(len(p_actions)))
        for num_of_trial, clicks_of_trial in enumerate(p_actions):
            p_num_actions[num_of_trial] = len(clicks_of_trial)

        algo_num_actions = pd.DataFrame.from_dict(algo_num_actions, orient='index')
        p_num_actions = pd.DataFrame.from_dict(p_num_actions, orient='index')
        if plot:
            plt.plot(algo_num_actions, label="Algorithm")
            plt.plot(p_num_actions, label="Participant")
            plt.xlabel('Trials')
            plt.ylabel('Number of clicks')
            plt.legend()
            #plt.show()
            plt.savefig(path, bbox_inches='tight')
            plt.close()
        return algo_num_actions, p_num_actions

    def plot_history(self, history, prior, obj_fn):
        # fig, ax = plt.subplots()
        # for t in range(0,history.max_t+1):
        #     df, w = history.get_distribution(m=0, t=t)
        #     pyabc.visualization.plot_kde_1d(df, w,
        #                                 x="", ax=ax,
        #                                 label="PDF t={}".format(t))
        # plt.legend()
        # plt.show()

        posterior = pyabc.transition.MultivariateNormalTransition()
        posterior.fit(*history.get_distribution(m=0))

        sim_prior_params = []
        sim_posterior_params = []
        prior_rewards = []
        posterior_rewards = []
        num_simulations = 100

        for i in range(num_simulations):
            prior_params = prior.rvs()
            sim_prior_params.append(prior_params)
            prior_sample = obj_fn(prior_params)
            prior_rewards.append(prior_sample["mer"][0])

            posterior_params = posterior.rvs()
            sim_posterior_params.append(posterior_params)
            posterior_sample = obj_fn(posterior_params)
            posterior_rewards.append(posterior_sample["mer"][0])

        mean_prior_rewards = np.mean(prior_rewards, axis=0)
        mean_posterior_rewards = np.mean(posterior_rewards, axis=0)
        plt.plot(mean_prior_rewards, label='Prior')
        plt.plot(mean_posterior_rewards, label='Posterior')
        plt.plot(self.participant.scores, label='Participant')
        plt.legend()
        plt.show()

        return history


def plot_model_selection_results(run_history, model_names):
    _, ax = plt.subplots(figsize=(10, 7))
    model_probs = run_history.get_model_probabilities()
    model_probs.columns = [model_names[c] for c in model_probs.columns]
    print(model_probs.to_string())
    ax = model_probs.plot.bar(legend=True, ax=ax)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Population index")
    plt.show()
    plt.show()


class BayesianModelSelection:
    def __init__(self, models_list, model_attributes, participant, env,
                 objective, num_simulations):
        self.optimizers = []
        self.models = []
        self.participant = participant
        self.env = env
        self.pipeline = env.pipeline
        self.objective = objective
        for i, model in enumerate(models_list):
            self.models.append(model)
            optimizer = ParameterOptimizer(model, model_attributes[i], participant, env)
            optimizer.num_simulations = num_simulations
            optimizer.objective = objective
            optimizer.optimizer = "pyabc"
            self.optimizers.append(optimizer)
        self.num_models = len(models_list)

    def model_selection(self):
        priors = []
        models = []
        for opt in self.optimizers:
            models.append(opt.objective_fn)
            priors.append(opt.get_prior())
        p_data = construct_p_data(self.participant, self.pipeline)
        observation = get_relevant_data(p_data, self.objective)
        distance_fn = construct_objective_fn("pyabc", self.objective, p_data, self.pipeline)
        transitions = [MultivariateNormalTransition(scaling=0.1) for _ in range(self.num_models)]
        abc = pyabc.ABCSMC(models, priors, distance_fn, transitions=transitions,
                           population_size=100)
        db_path = ("sqlite:///" + "test.db")
        abc.new(db_path, observation)
        history = abc.run(max_nr_populations=5)
        return history
