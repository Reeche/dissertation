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
    # exp_list = ['v1.0', 'c2.1', 'c1.1',
    #             'high_variance_high_cost',
    #             'high_variance_low_cost',
    #             'low_variance_high_cost',
    #             'low_variance_low_cost',
    #             'strategy_discovery'
    #             ]

    exp_list = ['v1.0']

    criterion = "likelihood"
    data_dir = f"../../results_mb_8000_v2/mcrl"

    for exp in exp_list:

        # if exp in ["v1.0", "c1.1", "c2.1"]:
        #     criterion = 'pseudo_likelihood'
        # else:
        #     criterion = 'number_of_clicks_likelihood'

        ##open existing df
        # df = pd.read_csv(f"{exp}.csv", index_col=0)

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

        for files in os.listdir(f"{data_dir}/{exp}_mb"):

            pid = int(files.split("_")[0])
            model_variant = files.split("_")[2].split(".")[0]
            data = pd.read_pickle(f"{data_dir}/{exp}_mb/{files}")

            p = E.participants[pid]
            participant_obj = ParticipantIterator(p)

            mf = ModelFitter(
                exp_name=exp,
                exp_attributes=exp_attributes,
                data_path=f"{data_dir}/{exp}_mb",
                number_of_trials=num_trials)

            pid_context, env = mf.get_participant_context(pid)

            pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

            if criterion == "likelihood":
                if model_variant == "uniform":
                    number_of_parameters = 3
                    data["sigma"] = 1
                elif model_variant == "linear":
                    number_of_parameters = 5
                elif model_variant == "full":
                    number_of_parameters = 7
                data["sigma"] = 1
            else:
                number_of_parameters = 4 #todo

            df.loc[len(df)] = [exp, pid, model_variant, data["a"][0], pid_context.clicks, data["mer"][0], pid_mer,
                               data["rewards"][0], pid_context.score,
                               click_loss(pid_context.clicks, data["a"], data),
                               mer_loss(pid_mer, data["mer"], data), data["loss"], number_of_parameters]

        df.to_csv(f"{exp}_mb_8000.csv")

