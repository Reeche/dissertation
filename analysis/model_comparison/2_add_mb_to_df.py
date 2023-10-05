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


def click_loss(p_data, model_data, model_params, exp):
    p_number_of_clicks_per_trial = [
        [len([click for click in p_clicks if click not in [0, None]]) for p_clicks in p_data]]
    a_number_of_clicks_per_trial = [
        [len([click for click in a_clicks if click not in [0, None]]) for a_clicks in algorithm_click_sequence] for
        algorithm_click_sequence in model_data]
    if exp != "mb":
        sigma = np.exp(model_params[0]["lik_sigma"])
    elif exp == "mb":
        sigma = np.exp(model_params[0]["sigma"])

    objective_value = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=np.exp(sigma))
            for x, y in zip(
            a_number_of_clicks_per_trial, p_number_of_clicks_per_trial
        )
        ]
    )
    return objective_value


def mer_loss(p_mer, model_data, model_params, exp):
    mean_mer = np.mean(model_data, axis=0)
    if exp != "mb":
        sigma = np.exp(model_params[0]["lik_sigma"])
    elif exp == "mb":
        sigma = np.exp(model_params[0]["sigma"])  # for mb
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


def mb_models(data_dir, exp, criterion):
    df = pd.DataFrame(
        columns=["exp", "criterion", "pid", "model", "model_clicks", "pid_clicks", "model_mer", "pid_mer",
                 "click_loss", "mer_loss", "loss"])
    # go to the mb models and open pkl
    for files in os.listdir(f"{data_dir}/{exp}_mb"):
        pid = int(files.split("_")[0])
        data = pd.read_pickle(f'{data_dir}/{exp}_mb/{pid}_likelihood.pkl') #todo: change back to criterion
        df.loc[len(df)] = [exp, criterion, pid, "mb", data["a"], None, data["mer"], None, None, None, data["loss"]]
    return df


if __name__ == "__main__":
    # exp_list = ['v1.0', 'c2.1', 'c1.1',
    #            'high_variance_high_cost',
    #            'high_variance_low_cost',
    #            'low_variance_high_cost',
    #            'low_variance_low_cost'
    #            ]
    exp_list = ['v1.0', 'low_variance_low_cost']

    iterations = 30

    data_dir = "../../results_vanilla_models/mcrl"

    for exp in exp_list:
        df = pd.DataFrame(
            columns=["exp", "criterion", "pid", "model", "model_clicks", "pid_clicks", "model_mer", "pid_mer",
                     "click_loss", "mer_loss", "loss"])

        if exp in ["v1.0", "c1.1", "c2.1"]:
            criterion = 'pseudo_likelihood'
        else:
            criterion = 'number_of_clicks_likelihood'

        E = Experiment(exp, data_path=f"../../results/cm/inferred_strategies/{exp}_training/")
        exp_attributes = {
            "exclude_trials": None,
            "block": None,
            "experiment": None,
            "click_cost": 1
        }

        for files in os.listdir(f"{data_dir}/{exp}_data"):
            pid = int(files.split("_")[0])
            model = int(files.split("_")[1])

            p = E.participants[pid]
            participant_obj = ParticipantIterator(p)

            mf = ModelFitter(
                exp_name=exp,
                exp_attributes=exp_attributes,
                data_path=f"{data_dir}/{exp}_mb",
                number_of_trials=35)

            pid_context, env = mf.get_participant_context(pid)

            pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

            data = pd.read_pickle(f'{data_dir}/{exp}_data/{pid}_{model}_{iterations}.pkl')
            model_params = pd.read_pickle(
                f'{data_dir}/{exp}_priors/{pid}_{criterion}_{model}.pkl')

            df.loc[len(df)] = [exp, criterion, pid, model, data["a"], pid_context.clicks, data["mer"], pid_mer,
                               click_loss(pid_context.clicks, data["a"], model_params[0], exp),
                               mer_loss(pid_mer, data["mer"], model_params[0], exp), click_sequence_loss(model_params)]

        # save df as csv
        mb_df = mb_models(data_dir, exp, criterion)
        df['pid_clicks'] = df.groupby('pid')['pid_clicks'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        df['pid_mer'] = df.groupby('pid')['pid_mer'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    df.to_csv(f"test.csv")

    # df.to_csv(f"{exp}_{criterion}.csv")
