import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import norm
from vars import pid_dict
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.env.modified_mouselab import get_termination_mers

"""
This script created csv for each condition and each model. 
Each csv contains the following columns:
- pid: participant id
- class: model class
- model_index: model index
- condition: condition
- loss: click sequence loss
- number_of_parameters: number of parameters
- model_clicks: model clicks
- model_mer: model mer
- model_rewards: model rewards
- pid_clicks: participant clicks
- pid_mer: participant mer
- pid_rewards: participant rewards
- click_loss: click loss
- mer_loss: mer loss
"""


def click_loss(p_data, model_data, criterion, model_params=None):
    p_number_of_clicks_per_trial = [
        [len([click for click in p_clicks if click not in [0, None]]) for p_clicks in p_data]]
    a_number_of_clicks_per_trial = [
        [len([click for click in a_clicks if click not in [0, None]]) for a_clicks in algorithm_click_sequence] for
        algorithm_click_sequence in model_data]

    if criterion == "likelihood":
        sigma = 1
    else:
        if model_params:
            sigma = np.exp(model_params[0]["lik_sigma"])
        else:
            sigma = 1

    objective_value = -np.sum(
        [
            norm.logpdf(x, loc=y, scale=sigma)
            for x, y in zip(
            a_number_of_clicks_per_trial, p_number_of_clicks_per_trial
        )
        ]
    )
    return objective_value


def mer_loss(p_mer, model_data, criterion, model_params=None):
    mean_mer = np.mean(model_data, axis=0)
    if criterion == "likelihood":
        sigma = 1
    else:
        if model_params:
            sigma = np.exp(model_params[0]["lik_sigma"])
        else:
            sigma = 1

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
    if model in [1756, 1743]:  # no learning and habitual
        if criterion == "likelihood":
            return 3
        else:
            return 4
    elif model in [527, 522]:  # RSSL
        if criterion == "likelihood":
            return 1
        else:
            return 2
    elif model in [491, 479, 486, 487, 490]:  # reinforce and lvoc
        if criterion == "likelihood":
            return 3
        else:
            return 4
    elif model in [480, 481]:
        if criterion == "likelihood":
            return 5
        else:
            return 6
    elif model in [482, 483, 484, 485, 488, 489]:
        if criterion == "likelihood":
            return 4
        else:
            return 5
    else:
        if criterion == "likelihood":
            return 8


def get_all_combinations(model_class, condition):
    # mapping = {"habitual": [1743], "non_learning": [1756], "hybrid": [491, 479], "ssl": [522], "pure": [491, 479],
    #            "mb": ["full"]}
    mapping = {"rl_hybrid_variants": [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490]}
    model_index = mapping[model_class]
    combinations = list(itertools.product([*pid_dict[condition]], [*model_index]))
    return combinations


exp_attributes = {
    "exclude_trials": None,
    "block": None,
    "experiment": None,
    "click_cost": 1
}

if __name__ == "__main__":
    root_folder = os.getcwd()
    # folder_list = ["habitual", "hybrid", "non_learning", "pure", "ssl", "mb"]
    # folder_list = ["non_learning", "pure", "ssl", "mb"]
    # conditions = ["v1.0", "c2.1", "c1.1",
    #               "high_variance_high_cost", "high_variance_low_cost",
    #               "low_variance_high_cost", "low_variance_low_cost",
    #               "strategy_discovery"]
    # conditions = ["v1.0", "c2.1", "c1.1"]
    folder_list = ["rl_hybrid_variants"]
    conditions = ["strategy_discovery"]

    for condition in conditions:

        if condition == "strategy_discovery":
            num_trials = 120
        else:
            num_trials = 35

        for model_class in folder_list:
            combinations = get_all_combinations(model_class, condition)

            # create new dataframe with the columns "pid", "class", model_index"
            df = pd.DataFrame(columns=["pid", "class", "model_index"])
            # class column are the second elements of the combinations
            df["model_index"] = [x[1] for x in combinations]
            # pid column are the first elements of the combinations
            df["pid"] = [x[0] for x in combinations]

            df["class"] = model_class
            df["condition"] = condition

            # iterate through the df rows and get pid and model
            for index, row in df.iterrows():
                pid = row["pid"]
                model = row["model_index"]
                print(pid, model)
                if model_class != "mb":
                    data = pd.read_pickle(f'{root_folder}/{model_class}/{condition}_data/{pid}_{model}_1.pkl')
                    model_params = pd.read_pickle(
                        f'{root_folder}/{model_class}/{condition}_priors/{pid}_likelihood_{model}.pkl')
                elif model_class == "mb":
                    data = pd.read_pickle(f'{root_folder}/{model_class}/{condition}_mb/{pid}_likelihood_full.pkl')
                else:  # print warning that model class is not recognized
                    print("Model class not recognized")

                # add loss to the row
                if model_class != "mb":
                    df.at[index, "loss"] = click_sequence_loss(model_params)
                elif model_class == "mb":
                    df.at[index, "loss"] = data["loss"]

                df.at[index, "number_of_parameters"] = number_of_parameters(model, criterion="likelihood")

                # add model information #todo: why str?
                df.at[index, "model_clicks"] = str(data["a"][0])
                df.at[index, "model_mer"] = str(data["mer"][0])
                if model_class != "mb":
                    df.at[index, "model_rewards"] = str(data["r"][0])
                elif model_class == "mb":
                    df.at[index, "model_rewards"] = str(data["rewards"][0])

                # add participant information
                mf = ModelFitter(
                    exp_name=condition,
                    exp_attributes=exp_attributes,
                    data_path=f"{model_class}/{condition}",
                    number_of_trials=num_trials)

                pid_context, env = mf.get_participant_context(pid)
                pid_mer = get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline)

                df.at[index, "pid_clicks"] = str(pid_context.clicks)
                df.at[index, "pid_mer"] = str(get_termination_mers(pid_context.envs, pid_context.clicks, env.pipeline))
                df.at[index, "pid_rewards"] = str(pid_context.score)

                if model_class != "mb":
                    df.at[index, "click_loss"] = click_loss(pid_context.clicks, data["a"], criterion="likelihood",
                                                            model_params=model_params[0])
                    df.at[index, "mer_loss"] = mer_loss(pid_mer, data["mer"], criterion="likelihood",
                                                        model_params=model_params[0])
                elif model_class == "mb":
                    df.at[index, "click_loss"] = click_loss(pid_context.clicks, data["a"], criterion="likelihood",
                                                            model_params=None)
                    df.at[index, "mer_loss"] = mer_loss(pid_mer, data["mer"], criterion="likelihood", model_params=None)

            # save the dataframe as csv
            df.to_csv(f"{model_class}_{condition}.csv")
            # print(df)
