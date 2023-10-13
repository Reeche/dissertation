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


def click_loss(p_data, model_data, model_params):
    p_number_of_clicks_per_trial = [
        [len([click for click in p_clicks if click not in [0, None]]) for p_clicks in p_data]]
    a_number_of_clicks_per_trial = [
        [len([click for click in a_clicks if click not in [0, None]]) for a_clicks in algorithm_click_sequence] for
        algorithm_click_sequence in model_data]
    objective_value = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=np.exp(np.exp(model_params["sigma"])))
            for x, y in zip(
            a_number_of_clicks_per_trial, p_number_of_clicks_per_trial
        )
        ]
    )
    return objective_value


def mer_loss(p_mer, model_data, model_params):
    mean_mer = np.mean(model_data, axis=0)
    normal_objective = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=np.exp(model_params["sigma"]))
            for x, y in zip(mean_mer, p_mer)
        ]
    )
    return normal_objective


if __name__ == "__main__":
    exp_list = ['v1.0', 'c2.1', 'c1.1',
               'high_variance_high_cost',
               'high_variance_low_cost',
               'low_variance_high_cost',
               'low_variance_low_cost'
               ]
    # exp_list = ['v1.0']

    iterations = 30

    data_dir = "../../results_vanilla_models/mcrl"

    for exp in exp_list:

        if exp in ["v1.0", "c1.1", "c2.1"]:
            criterion = 'pseudo_likelihood'
        else:
            criterion = 'number_of_clicks_likelihood'

        # open existing df
        df = pd.read_csv(f"{exp}_{criterion}.csv", index_col=0)

        E = Experiment(exp, data_path=f"../../results/cm/inferred_strategies/{exp}_training/")
        exp_attributes = {
            "exclude_trials": None,
            "block": None,
            "experiment": None,
            "click_cost": 1
        }

        for files in os.listdir(f"{data_dir}/{exp}_mb"):
            pid = int(files.split("_")[0])
            if files.split("_")[1] != "likelihood.pkl":
                data = pd.read_pickle(f'{data_dir}/{exp}_mb/{pid}_{criterion}.pkl')
            else:
                continue

            p = E.participants[pid]
            participant_obj = ParticipantIterator(p)

            mf = ModelFitter(
                exp_name=exp,
                exp_attributes=exp_attributes,
                data_path=f"{data_dir}/{exp}_mb",
                number_of_trials=35)

            pid_context, env = mf.get_participant_context(pid)

            pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

            if criterion != "likelihood":
                number_of_parameters = 4
            else:
                number_of_parameters = 5

            df.loc[len(df)] = [exp, criterion, pid, "mb", data["a"], pid_context.clicks, data["mer"], pid_mer,
                               click_loss(pid_context.clicks, data["a"], data),
                               mer_loss(pid_mer, data["mer"], data), data["loss"], number_of_parameters]

        df.to_csv(f"{exp}_{criterion}.csv")

    # df.to_csv(f"{exp}_{criterion}.csv")
