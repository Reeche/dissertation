import pandas as pd
import numpy as np
from scipy.stats import norm
import ast
import os

from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.env.modified_mouselab import get_termination_mers

"""
Create a df with the columns: model, pid, model clicks, pid clicks, model mer, pid mer, criterion, loss, number of clicks loss, pseudo likelihood loss
"""


def click_loss(p_data, model_data, model_params, criterion):
    p_number_of_clicks_per_trial = [
        [len([click for click in p_clicks if click not in [0, None]]) for p_clicks in p_data]]
    a_number_of_clicks_per_trial = [
        [len([click for click in a_clicks if click not in [0, None]]) for a_clicks in algorithm_click_sequence] for
        algorithm_click_sequence in model_data]

    if criterion == "likelihood":
        sigma = 1
    else:
        sigma = np.exp(model_params[0]["lik_sigma"])

    objective_value = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=sigma)
            for x, y in zip(
            a_number_of_clicks_per_trial, p_number_of_clicks_per_trial
        )
        ]
    )
    return objective_value


def mer_loss(p_mer, model_data, model_params, criterion):
    mean_mer = np.mean(model_data, axis=0)
    if criterion == "likelihood":
        sigma = 1
    else:
        sigma = np.exp(model_params[0]["lik_sigma"])
    normal_objective = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=sigma)
            for x, y in zip(mean_mer, p_mer)
        ]
    )
    return normal_objective


def click_sequence_loss(prior_data):
    losses = [trial["result"]["loss"] for trial in prior_data[0][1]]
    return min(np.absolute(losses))


def number_of_parameters(model, criterion):
    if model in [480, 481]: # PRW + SC
        return 5
    elif model in [482, 483, 484, 485, 488, 489]: #PRW or SC
        return 4
    elif model in [486, 487, 490, 491]: # vanilla, TD, PR only
        return 3


if __name__ == "__main__":
    # exp_list = ['v1.0',
    #             'c2.1',
    #             'c1.1',
    #             'high_variance_high_cost',
    #             'high_variance_low_cost',
    #             'low_variance_high_cost',
    #             'low_variance_low_cost',
    #             'strategy_discovery'
    #             ]

    exp_list = ['strategy_discovery']

    iterations = 1

    data_dir = "../../results_rl_variants_8000/mcrl"
    model_list = pd.read_csv("../../mcl_toolbox/models/rl_models.csv")

    for exp in exp_list:
        print(exp)
        df = pd.DataFrame(
            columns=["exp", "pid", "model", "model_clicks", "pid_clicks", "model_mer", "pid_mer", "model_rewards",
                     "pid_rewards", "click_loss", "mer_loss", "loss", "number_of_parameters"])

        E = Experiment(exp, data_path=f"../../results/cm/inferred_strategies/{exp}_training/")
        exp_attributes = {
            "exclude_trials": None,
            "block": None,
            "experiment": None,
            "click_cost": 1
        }

        if exp == "strategy_discovery":
            num_trials = 120
        else:
            num_trials = 35

        for files in os.listdir(f"{data_dir}/{exp}_data"):

            pid = int(files.split("_")[0])
            model = int(files.split("_")[1])
            if model in [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]:
                p = E.participants[pid]
                participant_obj = ParticipantIterator(p)

                mf = ModelFitter(
                    exp_name=exp,
                    exp_attributes=exp_attributes,
                    data_path=f"{data_dir}/{exp}",
                    number_of_trials=num_trials)

                pid_context, env = mf.get_participant_context(pid)

                pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

                data = pd.read_pickle(f'{data_dir}/{exp}_data/{pid}_{model}_{iterations}.pkl')
                model_params = pd.read_pickle(
                    f'{data_dir}/{exp}_priors/{pid}_likelihood_{model}.pkl')

                df.loc[len(df)] = [exp, pid, model, data["a"][0], pid_context.clicks, data["mer"][0], pid_mer,
                                   data["r"][0], pid_context.score,
                                   click_loss(pid_context.clicks, data["a"], model_params[0], criterion="likelihood"),
                                   mer_loss(pid_mer, data["mer"], model_params[0], criterion="likelihood"),
                                   click_sequence_loss(model_params),
                                   number_of_parameters(model, criterion="likelihood")]

        # save df as csv
        df.to_csv(f"{exp}.csv")
