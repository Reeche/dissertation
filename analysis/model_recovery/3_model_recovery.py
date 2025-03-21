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


def confusion_matrix(data):
    # Replace model names in "recovered_model" with the ones in rename_map
    data["recovered_model"] = data["recovered_model"].replace(rename_map)

    # Initialize confusion matrix with actual counts
    unique_models = data["recovered_model"].unique()
    confusion_matrix = pd.DataFrame(0, index=unique_models, columns=unique_models)

    for model in unique_models:
        # Filter for the current recovered model
        model_data = data[data["recovered_model"] == model]

        # Find the model with the lowest BIC for each participant
        min_bic_idx = model_data.groupby('pid')['BIC'].idxmin()
        lowest_bic_models = model_data.loc[min_bic_idx]

        # Count occurrences of each best-fitting model
        model_counts = lowest_bic_models['model'].value_counts()

        # Update the confusion matrix with actual counts
        for recovered_model, count in model_counts.items():
            confusion_matrix.loc[model, recovered_model] = count

    # Compute percentage confusion matrix (normalize rows)
    confusion_matrix_percentage = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100

    # replace NaN with 0
    confusion_matrix_percentage = confusion_matrix_percentage.fillna(0)
    confusion_matrix = confusion_matrix.fillna(0)

    # Extract model names (row/column labels)
    unique_models = confusion_matrix.index

    # Calculate R^2 for each model pair; something is wrong here but maybe also not needed?
    # r2_dict = {}
    # for model in unique_models:
    #     for recovered_model in unique_models:
    #         r = np.corrcoef(confusion_matrix.loc[model], confusion_matrix.loc[recovered_model])[0, 1]
    #         r2_dict[(model, recovered_model)] = r ** 2  # Square of correlation coefficient
    # print(r2_dict)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(confusion_matrix, cmap='Reds') #Wistia

    # Annotate each cell with count and percentage (fontsize = 12)
    for (i, j), val in np.ndenumerate(confusion_matrix.values):
        percentage = confusion_matrix_percentage.iloc[i, j]
        ax.text(j, i, f'{int(val)}\n({percentage:.2f}%)', ha='center', va='center', fontsize=12)

    # Set tick labels with larger font size
    plt.xticks(range(len(confusion_matrix.columns)), confusion_matrix.columns, rotation=45, fontsize=12, ha='left')
    plt.yticks(range(len(confusion_matrix.index)), confusion_matrix.index, fontsize=12)

    # **Adjust layout to avoid label cutoff**
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.8)
    # plt.savefig("confusion_matrix.png")
    # plt.show()

    return confusion_matrix, confusion_matrix_percentage




if __name__ == "__main__":
    # experiment = ["low_variance_high_cost"]
    # recovered_model = ["habitual"]
    recovered_model = ["hybrid_reinforce", "mf_reinforce", "habitual", "non_learning"]
    experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                  "low_variance_low_cost"]
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
    confusion_matrix(df_all)
