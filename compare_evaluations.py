import os
import pandas as pd
import sys
import ast

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from mcl_toolbox.utils.learning_utils import pickle_load

"""
This file contains analysis based on fitted mcrl models.
"""

def plot_loss_development(dir, loss_list):
    # loss development over all models
    average_loss = np.average(loss_list, axis=0)
    plt.plot(average_loss)
    # plt.show()
    plt.savefig(f"{dir}/mcrl/plots/loss_development.png")
    plt.close()

# create a dataframe of fitted models and pid
def create_dataframe_of_fitted_model_pid(pid_list, model_list, optimization_criterion, dir):
    df = pd.DataFrame(
        columns=["pid", "model", "loss", "BIC"])
    df["pid"] = sorted(pid_list * len(model_list))
    df["model"] = model_list * len(pid_list)

    temp_min_loss = []
    all_losses = []
    # temp_AIC = []
    temp_BIC = []
    for index, row in df.iterrows():
        try:
            prior_data = pd.read_pickle(
                f"{dir}/mcrl/{exp_num}_priors/{row['pid']}_{optimization_criterion}_{row['model']}.pkl")
            losses = [trial["result"]["loss"] for trial in prior_data[0][1]]
            all_losses.append(losses)
            min_loss = min(np.absolute(losses))
            temp_min_loss.append(min_loss)

            if row['model'] in ["447", "483"]:
                number_of_parameters = len(prior_data[1]) - 56
            else:
                # pr_weight is always added regardless of pr, therefore one more (unused) parameter need to be deducated
                number_of_parameters = len(prior_data[1]) - 57


            # min_loss is log-likelihood
            # AIC = 2 * min_loss + number_of_parameters * 2  # previous aic by Yash
            # temp_AIC.append(AIC)

            BIC = 2 * min_loss + number_of_parameters * np.log(120)
            temp_BIC.append(BIC)

        except Exception as e:
            # remove the row in df
            df = df.drop(index)
            print(f"{e} have been dropped")
            continue

    df["loss"] = temp_min_loss
    # df["AIC"] = temp_AIC
    df["BIC"] = temp_BIC

    ## sort by BIC
    df = df.sort_values(by=["BIC"])
    df["BIC"] = df["BIC"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    sorted_df = grouped_df.sort_values(by=["BIC"])
    print(sorted_df)

    plot_loss_development(dir, np.array(all_losses))

    return df


def average_performance_reward(
        exp_num, pid_list, optimization_criterion, model_index, dir, plotting=True
):
    """
    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num: a string "v1.0"
        pid_list: list of participant IDs
        optimization_criterion: a string
        model_index: an integer
        plotting: if true, plots are created

    Returns: average performance of model and participant

    """
    parent_directory = Path(__file__).parents[1]
    reward_info_directory = os.path.join(
        parent_directory, f"mcl_toolbox/{dir}/mcrl/{exp_num}_data"
    )

    data_model_temp = []
    # get reward of fitted models in list
    for pid in pid_list:
        try:
            data = pickle_load(
                os.path.join(
                    reward_info_directory,
                    f"{pid}_{model_index}_30.pkl",
                    # f"{pid}_{optimization_criterion}_{model_index}.pkl",
                )
            )
            data_model_temp.append(data["r"])
        except Exception as e:
            print(e)

        # data_model_temp.append(data["r"])

    # get participant rewards
    data_pid = pd.read_csv(f"data/human/{exp_num}/mouselab-mdp.csv")
    average_score = data_pid.groupby(["trial_index"]).mean()["score"]

    # create averaged values
    data_average = np.average(data_model_temp, axis=0)
    plt.plot(data_average[0], label=model_index)
    plt.plot(average_score, label="pid")
    plt.legend()
    plt.savefig(f"{dir}/mcrl/plots/{exp_num}_{model_index}.png")
    # plt.show()
    plt.close()

    return data_average


def create_bic_table(exp_num, pid_list, optimization_criterion, model_list):
    # df = create_dataframe_of_fitted_model_pid(pid_list, 35, optimization_criterion, model_list)
    df = df.drop(columns=['optimization_criterion', 'loss', 'AIC'])

    # drop not used models
    df = df[df['model'].isin(model_list)]

    df.to_csv(f"{exp_num}_bic.csv", index=False)

    df = df.reset_index()
    df = df.pivot_table(index=["pid"], columns=["model"], values=["BIC"], fill_value=0).reset_index()
    df.to_csv(f"matlab_{exp_num}_bic.csv")
    return None


if __name__ == "__main__":
    exp_num = 'c1.1'
    optimization_criterion = "pseudo_likelihood"
    hr = False
    if hr:
        model_list = ["272", "280", "2064", "2074"]
    else:
        # model_list = ["479", "1682", "491", "1744"]
        # model_list = ["447", "479", "483", "491"]
        model_list = ["491", "527"]

    pid_dict = {
        'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77,
                 80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
                 148, 150, 154, 155, 158, 160, 165, 169, 173],
        'c2.1': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
                 79, 84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138,
                 142, 145, 149, 152, 156, 162, 164, 166, 170, 172],
        'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
                 83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
                 151, 153, 157, 159, 161, 163, 167, 168, 171],
        'high_variance_high_cost': [0, 1, 10, 18, 22, 25, 30, 32, 38, 41, 46, 47, 49, 57, 60, 63, 65, 70, 74, 76, 81, 83,
                                    88, 89, 94, 103, 108, 109, 111, 114, 116, 118, 125, 129, 134, 139, 149, 150, 156, 161,
                                    164, 167, 169, 173, 177, 182, 188, 191, 195, 198, 199, 204],
        'high_variance_low_cost': [4, 7, 8, 17, 20, 23, 26, 29, 33, 35, 40, 48, 50, 51, 53, 56, 58, 64, 71, 78, 82, 87, 92,
                                   93, 95, 96, 101, 112, 117, 119, 122, 126, 131, 133, 136, 141, 145, 146, 151, 154, 158,
                                   162, 168, 175, 180, 185, 187, 189, 193, 197, 202, 205],
        'low_variance_high_cost': [2, 13, 14, 16, 21, 24, 28, 31, 36, 37, 43, 45, 54, 61, 62, 68, 69, 73, 79, 80, 84, 86,
                                   90, 97, 98, 100, 102, 107, 120, 124, 128, 132, 135, 138, 140, 144, 147, 153, 157, 160,
                                   163, 166, 171, 174, 181, 183, 192, 194, 201, 203, 206],
        'low_variance_low_cost': [3, 5, 6, 9, 11, 12, 15, 19, 27, 34, 39, 42, 44, 52, 55, 59, 66, 67, 72, 75, 77, 85, 91,
                                  99, 104, 105, 106, 110, 113, 115, 121, 123, 127, 130, 137, 142, 143, 148, 152, 155, 159,
                                  165, 170, 172, 176, 178, 179, 184, 186, 190, 196, 200, 207],
        'strategy_discovery': list(range(1, 57))}

    dir = "rssl_results"

    df = create_dataframe_of_fitted_model_pid(pid_dict[exp_num], model_list, optimization_criterion, dir)

    # for model_index in model_list:
    #     average_performance_reward(exp_num, pid_dict[exp_num], optimization_criterion, model_index, dir)
