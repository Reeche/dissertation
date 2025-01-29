import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import pymannkendall as mk
import ast
from statsmodels.formula.api import ols
import warnings

from vars import (process_data, process_clicks,
                  clicking_participants, learning_participants, assign_model_names, not_examined_all_pid,
                  adaptive_pid, mod_adaptive_pid, maladaptive_pid, examined_all_pid, habitual_examined_all_pid,
                  planningamount_nonlearners, planningamount_learners)

warnings.filterwarnings("ignore")

"""
Compare vanilla models based on the fit
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


def create_csv_for_matlab(data, exp, pid_filter):
    # create a csv file for matlab

    # filter for pid
    if pid_filter:
        data = data[data["pid"].isin(pid_filter)]

    data = data.pivot(index="model", columns="pid", values="BIC").T
    if exp != "strategy_discovery":
        # data = data[['Habitual', 'MB - Level, grouped', 'MB - Level, ind.',
        #              'MB - No assump., grouped', 'MB - No assump., ind.',
        #              'MB - Uniform, grouped', 'MB - Uniform, ind.',
        #              'Non-learning', 'SSL', 'hybrid LVOC', 'hybrid Reinforce', 'MF - LVOC', 'MF - Reinforce']]

        ## without non.learners
        data = data[['Habitual', 'Level, grouped', 'Level, ind.',
                     'No assump., grouped', 'No assump., ind.',
                     'Uniform, grouped', 'Uniform, ind.',
                     'SSL', 'hybrid LVOC', 'hybrid Reinforce', 'MF - LVOC', 'MF - Reinforce']]
    else:
        data = data[['Habitual', 'MB - Level, grouped', 'MB - Level, ind.',
                     'MB - No assump., grouped', 'MB - No assump., ind.',
                     'MB - Uniform, grouped', 'MB - Uniform, ind.',
                     'Non-learning', 'SSL', 'hybrid Reinforce', 'MF - Reinforce']]
    data.to_csv(f"matlab/{exp}_learners.csv", index=False, header=False)


def group_pid_by_bic(data, exp):
    # which model explains which participant best
    if exp != "strategy_discovery":
        data = data[data["class"].isin(["ssl", "hybrid", "pure", "habitual", "mb", "non_learning"])]
    else:
        data = data[data["class"].isin(["ssl", "hybrid", "mf", "habitual", "mb"])] #todo: for thesis
        # data = data[data["class"].isin(["ssl", "hybrid", "mf", "habitual"])]  # todo: for CogSci

    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]

    # for each class, get the list of unique pids
    for model in data["class"].unique():
        print(f"{model}: {res[res['class'] == model]['pid'].unique()}")
    return res


def group_pid_by_score(data):
    ## optional filter only for 1743, 491, 479 (1756 is not learning)
    data = data[data["model"].isin([1743, 1756, 491, 479, 522])]

    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['reward_loss'].idxmin()
    res = data.loc[min_bic_idx]
    return res


def plot_pid_score_grouped_by_model(data, condition=None):
    # plot the score of the participants who are best explained by a certain model
    models = list(data["model"].unique())

    for model in models:
        # filter for the model in data
        filtered_data = data[data["model"] == model]
        # print(len(filtered_data), "unique pid are best explained by the model", model)
        if len(filtered_data) == 0:
            continue
        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(
            lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)

        # Mann Kendall test of trend
        # mk_results = mk.original_test(pid_rewards_average)
        # print(f"{condition}, {model}: {mk_results}")

        # plot the average
        plt.plot(pid_rewards_average, label=f"{model}, N={len(filtered_data)}")

        # add 95% confidence interval
        # plt.fill_between(np.arange(1, 36), pid_rewards_average - 1.96 * np.std(pid_rewards, axis=0),
        #                  pid_rewards_average + 1.96 * np.std(pid_rewards, axis=0), alpha=0.2)

        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend(fontsize=9, ncol=2)

    # save the plot
    plt.show()

    ##if no dir, create dir
    # if not os.path.exists(f"plots/{exp}"):
    #     os.makedirs(f"plots/{exp}")

    # plt.savefig(f"plots/{condition}/score_grouped.png")
    plt.close()

    return None


def plot_pid_clicks_grouped_by_model(data, condition):
    # plot the clicks of the participants who are best explained by a certain model

    # get unique models
    models = list(data["model"].unique())

    plt.figure(figsize=(8, 6))

    for model in models:
        # filter for the model in data
        filtered_data = data[data["model"] == model]
        # print(len(filtered_data), "unique pid are best explained by the model", model)
        if len(filtered_data) == 0:
            continue

        filtered_data['pid_clicks'] = filtered_data['pid_clicks'].apply(lambda x: ast.literal_eval(x))

        # calculate the average of the pid_clicks
        pid_clicks = np.array(filtered_data["pid_clicks"].to_list())
        result_array = np.array([[len(cell) - 1 for cell in row] for row in pid_clicks])
        pid_clicks_average = np.mean(result_array, axis=0)

        # print the average number of clicks and std
        print(f"{model} pid mean: {np.mean(pid_clicks_average)}")
        print(f"{model} pid std: {np.std(pid_clicks_average)}")

        # Mann Kendall test of trend
        # mk_results = mk.original_test(pid_clicks_average)
        # print(f"{condition}, {model}: {mk_results}")

        # plot the average
        plt.plot(pid_clicks_average, label=f"{model}, N={len(filtered_data)}")

        if condition == "high_variance_low_cost":
            # plt.axhline(y=7.10, color='r', linestyle='-')
            plt.ylim(-8, 13)
        elif condition == "high_variance_high_cost":
            # plt.axhline(y=6.32, color='r', linestyle='-')
            plt.ylim(-8, 13)
        elif condition == "low_variance_high_cost":
            # plt.axhline(y=0, color='r', linestyle='-')
            plt.ylim(-1, 10)
        else:
            # plt.axhline(y=5.82, color='r', linestyle='-')
            plt.ylim(-1, 10)

        # add 95% confidence interval
        # plt.fill_between(np.arange(1, 36), pid_clicks_average - 1.96 * np.std(result_array, axis=0),
        #                  pid_clicks_average + 1.96 * np.std(result_array, axis=0), alpha=0.2)

        # plt.title(label)
        plt.xlabel("Trial", fontsize=12)
        plt.ylabel("Average number of clicks", fontsize=12)
        plt.legend(fontsize=11, ncol=2)

    ##if no dir, create dir
    # if not os.path.exists(f"plots/{exp}"):
    #     os.makedirs(f"plots/{exp}")

    # plt.savefig(f"plots/{exp}/clicks_grouped.png")
    # plt.show()
    # plt.close()

    return None


def plot_model_simulation_of_pid_with_lowest_BIC(data, criteria):
    for model in data["model_index"].unique():
        filtered_data = data[data["model_index"] == model]
        filtered_data = process_data(filtered_data, f"model_{criteria}", f"pid_{criteria}", exp)

        if criteria == "rewards":
            filtered_data[f'pid_{criteria}'] = filtered_data[f'pid_{criteria}'].apply(
                lambda s: [float(num) for num in s.strip('[]').split()])
            filtered_data[f'model_{criteria}'] = filtered_data[f'model_{criteria}'].apply(lambda x: ast.literal_eval(x))
        elif criteria == "clicks":
            filtered_data["pid_clicks"] = filtered_data["pid_clicks"].apply(process_clicks)
            filtered_data["model_clicks"] = filtered_data["model_clicks"].apply(process_clicks)

        ### PID data
        pid_values = np.array(filtered_data[f"pid_{criteria}"].to_list())
        pid_values_average = np.mean(pid_values, axis=0)

        plt.plot(pid_values_average, label=f"Participant, N={len(filtered_data)}")
        # add 95%CI
        plt.fill_between(np.arange(0, 35), pid_values_average - 1.96 * np.std(pid_values, axis=0),
                         pid_values_average + 1.96 * np.std(pid_values, axis=0), alpha=0.2)

        ### model data
        model_data = np.array(filtered_data[f"model_{criteria}"].to_list())
        model_data_average = np.mean(model_data, axis=0)

        # rename model_index to model
        model = filtered_data["model"].unique()[0]
        plt.plot(model_data_average, label=f"{model}, N={len(filtered_data)}")

        plt.xlabel("Trial")
        plt.ylabel(f"Average {criteria}")
        plt.legend(fontsize=12)

        plt.show()
        plt.close()

    return None


def kruskal_rewards(exp, data):
    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_rewards"]]

    # convert pid_rewards to list
    data['pid_rewards'] = data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

    # if exp == "strategy_discovery":
    #     data["pid_rewards"] = data["pid_rewards"].apply(lambda x: x[-60:])
    # else:
    #     data["pid_rewards"] = data["pid_rewards"].apply(lambda x: x[-10:])

    # for each pid_rewards, calculate the average of the last 60 trials
    # if exp == "strategy_discovery":
    #     data["pid_rewards"] = data["pid_rewards"].apply(lambda x: np.mean(x[-60:]))
    # else:
    #     data["pid_rewards"] = data["pid_rewards"].apply(lambda x: np.mean(x[-10:]))

    # groupby model and conduct anova using statsmodel
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    #
    # model = ols('pid_rewards ~ model', data=data).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

    # remove all the model that does not contain Reinforce or Habitual
    # data = data[data["model"].str.contains("Reinforce") | data["model"].str.contains("Habitual")]

    # group the MB models and MF models together; hybrid and MF are grouped together as "MF"
    # data["model"] = data["model"].apply(lambda x: "MB" if "MB" in x else x)
    # data["model"] = data["model"].apply(lambda x: "Reinforce" if "MF" in x else x)
    # data["model"] = data["model"].apply(lambda x: "MF" if "LVOC" in x else x)
    # data["model"] = data["model"].apply(lambda x: "Reinforce" if "hybrid" in x else x)

    # group by model and get the pid_rewards
    grouped = data.groupby("model")["pid_rewards"].apply(list)
    res = kruskal([item for sublist in grouped[0] for item in sublist],
                  [item for sublist in grouped[1] for item in sublist],
                  [item for sublist in grouped[2] for item in sublist])
    print(f"Kruskal between all the models for {exp}: {res}")

    model_list = data["model"].unique()
    for model in model_list:
        model_data = data[data["model"] == model]["pid_rewards"]
        for model2 in model_list:
            if model == model2:
                print(f"{model} Mean: {np.mean(list(model_data))}")
                print(f"{model} Std: {np.std(list(model_data))}")
                continue
            model2_data = data[data["model"] == model2]["pid_rewards"]

            # res = mannwhitneyu(model_data, model2_data, alternative="two-sided")
            res = mannwhitneyu([item for sublist in list(model_data) for item in sublist],
                               [item for sublist in list(model2_data) for item in sublist], alternative="two-sided")
            print(f"Mann Whitney U {model}, {model2}: ", res)
    return None


def kruskal_clicks(exp, data):
    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_clicks"]]

    # convert pid_clicks to list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: ast.literal_eval(x))
    # get length of each list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: [len(cell) - 1 for cell in x])

    # for each pid_rewards, keep only the last 10 trials
    # if exp == "strategy_discovery":
    #     data["pid_clicks"] = data["pid_clicks"].apply(lambda x: x[-60:])
    # else:
    #     data["pid_clicks"] = data["pid_clicks"].apply(lambda x: x[-10:])

    # for each pid_rewards, calculate the average of the last 60 trials
    # if exp == "strategy_discovery":
    #     data["pid_clicks"] = data["pid_clicks"].apply(lambda x: np.mean(x[-60:]))
    # else:
    #     data["pid_clicks"] = data["pid_clicks"].apply(lambda x: np.mean(x[-10:]))

    # group the MF models together
    # data["model"] = data["model"].apply(lambda x: "MB" if "MB" in x else x)
    # data["model"] = data["model"].apply(lambda x: "MF" if "Reinforce" in x else x)
    # data["model"] = data["model"].apply(lambda x: "MF" if "LVOC" in x else x)

    # for data, replace all "model" by MCRL if not Habitual
    # This is for testing the difference between Habitual vs MCRL participants
    # There are no SSL participants in the data
    data["model"] = data["model"].apply(lambda x: "MCRL" if "Habitual" not in x else x)

    # group by model and get the pid_rewards
    grouped = data.groupby("model")["pid_clicks"].apply(list)
    # flatten the list
    # res = kruskal([item for sublist in grouped[0] for item in sublist],
    #               [item for sublist in grouped[1] for item in sublist],
    #               [item for sublist in grouped[2] for item in sublist])
    # print(f"Kruskal between all the models for {exp}: {res}")

    model_list = data["model"].unique()
    for model in model_list:
        model_data = data[data["model"] == model]["pid_clicks"]
        for model2 in model_list:
            if model == model2:
                ### calculate the average and std
                print(f"{model} mean: {np.mean(list(model_data))}")
                print(f"{model} std: {np.std(list(model_data))}")
                continue
            model2_data = data[data["model"] == model2]["pid_clicks"]

            res = mannwhitneyu([item for sublist in list(model_data) for item in sublist],
                               [item for sublist in list(model2_data) for item in sublist], alternative="two-sided")
            print(f"Mann Whitney U {model}, {model2}: ", res)

    return None


def linear_regression_clicks(data, exp):
    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_clicks"]]

    # convert pid_clicks to list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: ast.literal_eval(x))
    # get length of each list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: [len(cell) - 1 for cell in x])

    # get length of the dataframe before explode
    len_df = len(data)

    # explode the pid_clicks
    data = data.explode('pid_clicks')
    data = data.reset_index(drop=True)

    # add "trial" column that is a list from 1 to 35 * len_df
    data["trial"] = list(range(1, 36)) * len_df

    # res = ols('score ~ trial*C(condition, Treatment("mf"))', data=exp_score_data).fit()
    # print(res.summary())

    # make pid_clicks as int
    data['pid_clicks'] = data['pid_clicks'].astype(int)

    res = ols('pid_clicks ~ trial*C(model, Treatment("hybrid Reinforce"))', data=data).fit()
    print(res.summary())

    return None


if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    # experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
    #               "low_variance_low_cost"]
    experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                  "low_variance_low_cost"]
    # experiment = ["low_variance_low_cost"]

    for exp in experiment:
        print(exp)
        df_all = []
        data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)

        if exp in ["v1.0", "c1.1", "c2.1"]:
            data = data[data["pid"].isin(clicking_participants[exp])]
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                     "low_variance_low_cost"]:
            # data = data[data["pid"].isin(learning_participants[exp])]
            data = data[data["pid"].isin(planningamount_learners)]
        elif exp == "strategy_discovery":
            data = data[data["pid"].isin(clicking_participants[exp])]
            # data = data[data["pid"].isin(not_examined_all_pid)]

        # create a new column. If column "class" = "hybrid" and "model_index" = 491, then "model" = "pure Reinforce"
        data['model'] = data.apply(assign_model_names, axis=1)

        # remove all the MB models
        # data = data[~data["model"].str.contains("MB")]

        # calculate_score_loss(data)

        if exp == "strategy_discovery":
            data["BIC"] = compare_loss(data, 120)
        else:
            data["BIC"] = compare_loss(data, 35)

        df_all.append(data)

        ### for individual analysis
        result_df = pd.concat(df_all, ignore_index=True)

        create_csv_for_matlab(result_df, exp, planningamount_learners)

        res = group_pid_by_bic(result_df, exp)

        # print which of the pid is best explained by which model
        # for model in res["model"].unique():
        #     print(f"{model}: {res[res['model'] == model]['pid'].unique()}")

        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            plot_model_simulation_of_pid_with_lowest_BIC(res, "mer")
        #     # plot_pid_score_grouped_by_model(res, exp)
        #     # kruskal_rewards(exp, res)
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                     "low_variance_low_cost"]:
            # plot_model_simulation_of_pid_with_lowest_BIC(res, "clicks")
            # plot_pid_clicks_grouped_by_model(res, exp)
            # linear_regression_clicks(res, exp)
            # kruskal_rewards(exp, res)
            kruskal_clicks(exp, res)

    # result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "strategy_discovery_discovered_pid")
    # model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
    # res = group_pid_by_bic(result_df, exp)
    # plot_pid_score_grouped_by_model(res)
