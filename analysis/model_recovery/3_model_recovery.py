import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import pymannkendall as mk
import ast
from statsmodels.formula.api import ols
import warnings

from vars import assign_model_names, rename_map

warnings.filterwarnings("ignore")

"""
For the model recovery, compare the simulated models' performance against its fit. 
That is, I fitted the models to the simulated model's data. 
Then create confusion matrix, which of the model fits best to the model simulations. 
"""


def compare_clicks_likelihood(data, trials):
    BIC = 2 * data["click_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


def compare_mer_likelihood(data, trials):
    BIC = 2 * data["mer_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


def calculate_score_loss(data):
    # for every row, model_rewards - pid_rewards
    data['model_rewards'] = data['model_rewards'].apply(ast.literal_eval)
    data['pid_rewards'] = data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])
    pid_rewards_array = np.array(data['pid_rewards'].to_list(), dtype=np.float32)
    model_rewards_array = np.array(data['model_rewards'].to_list(), dtype=np.float32)
    reward_loss_array = pid_rewards_array - model_rewards_array
    data['reward_difference'] = reward_loss_array.tolist()

    # sum over the absolute values of reward_difference for each row
    data['reward_loss'] = data['reward_difference'].apply(lambda x: np.sum(np.abs(x)))
    return data


def compare_loss(data, trials):
    BIC = 2 * data["loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


def sort_by_BIC(data):
    df = data.sort_values(by=["BIC"])
    average_bic = df.groupby('model')['BIC'].mean().reset_index()
    sorted_df = average_bic.sort_values(by='BIC', ascending=True)
    # the smaller BIC the better
    print(sorted_df)
    return sorted_df


def group_pid_by_bic(data):
    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]
    return res


def confusion_matrix(data, rename_map=None, plot=True):
    # Custom model order
    model_order = ["hybrid Reinforce", "MF - Reinforce", "Habitual", "Non-learning"]

    if rename_map:
        data["recovered_model"] = data["recovered_model"].replace(rename_map)

    # Ensure all models in data are included in the order list
    all_models = set(data["recovered_model"].unique()).union(set(data["model"].unique()))
    missing_models = all_models - set(model_order)
    if missing_models:
        model_order.extend(sorted(missing_models))  # Append any missing models at the end

    # Initialize confusion matrix
    confusion_matrix = pd.DataFrame(0, index=model_order, columns=model_order)

    for model in model_order:
        model_data = data[data["recovered_model"] == model]
        min_bic_idx = model_data.groupby('pid')['BIC'].idxmin()
        lowest_bic_models = model_data.loc[min_bic_idx]
        model_counts = lowest_bic_models['model'].value_counts()

        for recovered_model, count in model_counts.items():
            confusion_matrix.loc[model, recovered_model] = count

    confusion_matrix = confusion_matrix.fillna(0)

    # Posterior likelihood: P(True | Recovered)
    posterior = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100
    posterior = posterior.fillna(0)

    # Inverse likelihood: P(Recovered | True)
    inverse = posterior.div(posterior.sum(axis=0), axis=1) * 100
    inverse = inverse.fillna(0)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, matrix, title in zip(axes, [posterior, inverse], ["Posterior Likelihood", "Inverse Likelihood"]):
            cax = ax.matshow(matrix.loc[model_order, model_order], cmap='Reds')
            for (i, j), val in np.ndenumerate(confusion_matrix.loc[model_order, model_order].values):
                percentage = matrix.loc[model_order, model_order].iloc[i, j]
                if title == "Posterior Likelihood":
                    count = confusion_matrix.loc[model_order, model_order].iloc[i, j]
                    label = f'{int(count)}\n({percentage:.2f}%)'
                else:
                    label = f'{percentage:.2f}%'
                ax.text(j, i, label, ha='center', va='center', fontsize=12)
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels(model_order, rotation=45, fontsize=12, ha='left')
            ax.set_yticks(range(len(model_order)))
            ax.set_yticklabels(model_order, fontsize=12)
            ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig("results/sd_model_recovery.png")
        plt.show()
        plt.close()

    return confusion_matrix, posterior, inverse


if __name__ == "__main__":
    experiment = ["strategy_discovery"]
    # recovered_model = ["habitual"]
    recovered_model = ["hybrid_reinforce", "mf_reinforce", "habitual", "non_learning"]
    # experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
    #               "low_variance_low_cost"]
    df_all = pd.DataFrame()
    for exp in experiment:
        for model in recovered_model:
            data = pd.read_csv(f"data/{model}_{exp}.csv", index_col=0)

            data['model'] = data.apply(assign_model_names, axis=1)
            data["recovered_model"] = model

            if exp == "strategy_discovery":
                data["BIC"] = compare_loss(data, 120)
            else:
                data["BIC"] = compare_loss(data, 35)

            res = group_pid_by_bic(data)

            # model_bic = sort_by_BIC(result_df)

            # print which of the pid is best explained by which model
            # for model in res["model"].unique():
            #     print(f"{model}: {res[res['model'] == model]['pid'].unique()}")

            # add res to df_all
            df_all = pd.concat([df_all, res])

    # create confusion matrix
    confusion_matrix(df_all, rename_map, plot=True)
