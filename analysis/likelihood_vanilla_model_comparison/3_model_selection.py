import pandas as pd
import numpy as np
import os
from vars import clicking_participants, learning_participants, discovered_participants
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import pymannkendall as mk
import ast
import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings

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


def bms(model_bic):
    # Step 1: Calculate BIC for each model
    # Assume bic_values is a list of pre-calculated BIC values for different models

    # Step 2: Compute the BIC differences
    delta_bic = np.array(model_bic["BIC"]) - min(model_bic["BIC"])

    # Step 3: Calculate exceedance probabilities
    exceedance_probs = np.exp(-0.5 * delta_bic) / np.sum(np.exp(-0.5 * delta_bic))

    # Step 4: Calculate φ (phi)
    phi = 1 / (1 + np.exp(delta_bic / 2))

    # Print results
    rounded_probs = [round(prob, 4) for prob in exceedance_probs]
    phi = [round(value, 4) for value in phi]
    print("Exceedance Probabilities:", rounded_probs)
    print("Phi Values:", phi)


def create_csv_for_matlab(data, exp):
    # create csv for matlab; filter for required models
    # data = data[data["model"].isin(["1743", "1756", "479", "491", "522", "full"])]
    # data = data[data["model"].isin([1743, 1756, 479, 491, 522])]
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    # habitual, MB, non-learning, SSL, hybrid LVOC, hybrid Reinforce, pure LVOC, pure Reinforce
    data.to_csv(f"{exp}.csv", index=False, header=False)


def group_pid_by_bic(data):
    # which model explains which participant best
    # data = data[data["class"].isin(["ssl", "hybrid", "pure", "habitual", "mb", "non_learning"])]
    data = data[data["class"].isin(["ssl", "hybrid", "pure", "habitual", "mb"])]

    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]
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
        print(len(filtered_data), "unique pid are best explained by the model", model)
        if len(filtered_data) == 0:
            continue
        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(
            lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)

        # plot the average
        plt.plot(pid_rewards_average, label=f"{model}, N={len(filtered_data)}")

        # add 95% confidence interval
        # plt.fill_between(np.arange(1, 36), pid_rewards_average - 1.96 * np.std(pid_rewards, axis=0),
        #                  pid_rewards_average + 1.96 * np.std(pid_rewards, axis=0), alpha=0.2)

        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend()

    # save the plot
    # plt.show()

    ##if no dir, create dir
    if not os.path.exists(f"plots/{exp}"):
        os.makedirs(f"plots/{exp}")

    plt.savefig(f"plots/{condition}/score_grouped.png")
    plt.close()

    return None


def plot_pid_clicks_grouped_by_model(data, condition):
    # plot the clicks of the participants who are best explained by a certain model

    # get unique models
    models = list(data["model"].unique())

    # for model_type, models in model_dict.items():
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

        # Mann Kendall test of trend
        # mk_results = mk.original_test(pid_clicks_average)
        # print(f"{condition}, {model}: {mk_results}")

        # plot the average
        plt.plot(pid_clicks_average, label=f"{model}, N={len(filtered_data)}")

        if condition == "high_variance_low_cost":
            label = "High variance low cost"
            plt.axhline(y=7.10, color='r', linestyle='-')
        elif condition == "high_variance_high_cost":
            label = "High variance high cost"
            plt.axhline(y=6.32, color='r', linestyle='-')
        elif condition == "low_variance_high_cost":
            label = "Low variance high cost"
            plt.axhline(y=0.01, color='r', linestyle='-')  # it is actually 0 but needs to show on plot, therefore 0.01
        else:
            label = "Low variance low cost"
            plt.axhline(y=5.82, color='r', linestyle='-')

        # add 95% confidence interval
        # plt.fill_between(np.arange(1, 36), pid_clicks_average - 1.96 * np.std(result_array, axis=0),
        #                  pid_clicks_average + 1.96 * np.std(result_array, axis=0), alpha=0.2)

        plt.title(label)
        plt.xlabel("Trial")
        plt.ylabel("Average number of clicks")
        plt.legend()

    plt.show()

    # if no dir, create dir
    # if not os.path.exists(f"plots/{exp}"):
    #     os.makedirs(f"plots/{exp}")

    # plt.savefig(f"plots/{exp}/clicks_grouped.png")
    plt.close()

    return None


def kruskal_rewards(exp, data):
    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_rewards"]]

    # convert pid_rewards to list
    data['pid_rewards'] = data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

    # for each pid_rewards, calculate the average of the last 60 trials
    if exp == "strategy_discovery":
        data["pid_rewards"] = data["pid_rewards"].apply(lambda x: np.mean(x[-60:]))
    else:
        data["pid_rewards"] = data["pid_rewards"].apply(lambda x: np.mean(x[-10:]))

    # groupby model and conduct anova using statsmodel
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    #
    # model = ols('pid_rewards ~ model', data=data).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

    ## Kruskal-Wallis H-test
    habitual = data[data["model"] == "Habitual"]["pid_rewards"]
    pure_reinforce = data[data["model"] == "pure Reinforce"]["pid_rewards"]
    pure_lvoc = data[data["model"] == "pure LVOC"]["pid_rewards"]
    hybrid_reinforce = data[data["model"] == "hybrid Reinforce"]["pid_rewards"]
    hybrid_lvoc = data[data["model"] == "hybrid LVOC"]["pid_rewards"]
    # from scipy import stats
    # res = stats.kruskal(q, x, y, z, a)
    # print(res)

    # combine the pure_reinforce and pure_lvoc
    pure_models = np.concatenate((pure_reinforce, pure_lvoc))
    hybrid_models = np.concatenate((hybrid_reinforce, hybrid_lvoc))
    reinforce_models = np.concatenate((pure_reinforce, hybrid_reinforce))
    lvoc_models = np.concatenate((pure_lvoc, hybrid_lvoc))

    from scipy import stats
    res = stats.kruskal(pure_models, hybrid_models, habitual)
    print("Kruskal pure, hybrid, habitual: ", res)

    res = mannwhitneyu(pure_models, habitual, alternative="two-sided")
    print("Pure models vs habitual: ", res)

    res = mannwhitneyu(hybrid_models, habitual, alternative="two-sided")
    print("Hybrid models vs habitual: ", res)

    res = mannwhitneyu(hybrid_models, pure_models, alternative="two-sided")
    print("Hybrid models vs pure model: ", res)

    ## between reinforce, lvoc, habitual
    res = stats.kruskal(reinforce_models, lvoc_models, habitual)
    print("Kruskal between reinforce, lvoc, habitual: ", res)

    res = mannwhitneyu(reinforce_models, habitual, alternative="two-sided")
    print("Reinforce models vs habitual: ", res)

    res = mannwhitneyu(lvoc_models, habitual, alternative="two-sided")
    print("LVOC models vs habitual: ", res)

    res = mannwhitneyu(reinforce_models, lvoc_models, alternative="two-sided")
    print("Reinforce vs LVOC: ", res)

    return None


def kruskal_clicks(exp, data):
    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_clicks"]]

    # convert pid_clicks to list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: ast.literal_eval(x))
    # get length of each list
    data['pid_clicks'] = data['pid_clicks'].apply(lambda x: [len(cell) - 1 for cell in x])

    # for each pid_rewards, calculate the average of the last 60 trials
    if exp == "strategy_discovery":
        data["pid_clicks"] = data["pid_clicks"].apply(lambda x: np.mean(x[-60:]))
    else:
        data["pid_clicks"] = data["pid_clicks"].apply(lambda x: np.mean(x[-10:]))

    ## Kruskal-Wallis H-test
    habitual = data[data["model"] == "Habitual"]["pid_clicks"]
    pure_reinforce = data[data["model"] == "pure Reinforce"]["pid_clicks"]
    pure_lvoc = data[data["model"] == "pure LVOC"]["pid_clicks"]
    hybrid_reinforce = data[data["model"] == "hybrid Reinforce"]["pid_clicks"]
    hybrid_lvoc = data[data["model"] == "hybrid LVOC"]["pid_clicks"]
    # non_learning = data[data["model"] == "Non-learning"]["pid_clicks"]

    # combine the pure_reinforce and pure_lvoc
    pure_models = np.concatenate((pure_reinforce, pure_lvoc))
    hybrid_models = np.concatenate((hybrid_reinforce, hybrid_lvoc))
    reinforce_models = np.concatenate((pure_reinforce, hybrid_reinforce))
    lvoc_models = np.concatenate((pure_lvoc, hybrid_lvoc))

    from scipy import stats
    res = stats.kruskal(pure_models, hybrid_models, habitual)
    print("Kruskal pure, hybrid, habitual: ", res)

    res = mannwhitneyu(pure_models, habitual, alternative="two-sided")
    print("Pure models vs habitual: ", res)

    res = mannwhitneyu(hybrid_models, habitual, alternative="two-sided")
    print("Hybrid models vs habitual: ", res)

    res = mannwhitneyu(hybrid_models, pure_models, alternative="two-sided")
    print("Hybrid models vs pure model: ", res)

    # res = mannwhitneyu(non_learning, pure_models, alternative="two-sided")
    # print("Hybrid models vs pure model: ", res)

    # res = mannwhitneyu(non_learning, hybrid_models, alternative="two-sided")
    # print("Hybrid models vs pure model: ", res)

    # res = mannwhitneyu(non_learning, habitual, alternative="two-sided")
    # print("Hybrid models vs pure model: ", res)

    ## between reinforce, lvoc, habitual
    # res = stats.kruskal(reinforce_models, lvoc_models, habitual, non_learning)
    res = stats.kruskal(reinforce_models, lvoc_models, habitual)
    print("Kruskal between reinforce, lvoc, habitual: ", res)

    res = mannwhitneyu(reinforce_models, habitual, alternative="two-sided")
    print("Reinforce models vs habitual: ", res)

    res = mannwhitneyu(lvoc_models, habitual, alternative="two-sided")
    print("LVOC models vs habitual: ", res)

    res = mannwhitneyu(reinforce_models, lvoc_models, alternative="two-sided")
    print("Reinforce vs LVOC: ", res)

    # res = mannwhitneyu(non_learning, lvoc_models, alternative="two-sided")
    # print("Non_learning vs LVOC: ", res)

    # res = mannwhitneyu(non_learning, reinforce_models, alternative="two-sided")
    # print("Non_learning vs Reinforce: ", res)

    # res = mannwhitneyu(non_learning, habitual, alternative="two-sided")
    # print("Non_learning vs Habitual: ", res)

    print("-------------------")

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


def assign_model_names(row):
    if row['class'] == 'hybrid' and row['model_index'] == "491":
        return 'hybrid Reinforce'
    elif row['class'] == 'hybrid' and row['model_index'] == "479":
        return 'hybrid LVOC'
    elif row['class'] == 'pure' and row['model_index'] == "491":
        return 'pure Reinforce'
    elif row['class'] == 'pure' and row['model_index'] == "479":
        return 'pure LVOC'
    elif row['model_index'] == "1743":
        return 'Habitual'
    elif row['model_index'] == "1756":
        return 'Non-learning'
    elif row['model_index'] == "522":
        return 'SSL'
    elif row['model_index'] == "full":
        return 'Model-based'
    else:
        raise ValueError("Model class combination not found")


if __name__ == "__main__":
    experiment = ["v1.0", "c2.1", "c1.1"]
    # experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
    #               "low_variance_low_cost"]
    # experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
    #               "low_variance_low_cost"]
    # experiment = ["c1.1"]
    # df_all = []
    for exp in experiment:
        df_all = []
        data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)

        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            data = data[data["pid"].isin(clicking_participants[exp])]
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                     "low_variance_low_cost"]:
            data = data[data["pid"].isin(learning_participants[exp])]

        # create a new column. If column "class" = "hybrid" and "model_index" = 491, then "model" = "pure Reinforce"
        data['model'] = data.apply(assign_model_names, axis=1)

        # calculate_score_loss(data)

        if exp == "strategy_discovery":
            data["BIC"] = compare_loss(data, 120)
        else:
            data["BIC"] = compare_loss(data, 35)

        df_all.append(data)

        ### for individual analysis
        result_df = pd.concat(df_all, ignore_index=True)

        # create_csv_for_matlab(result_df, exp)

        res = group_pid_by_bic(result_df)  # get BIC for only selected models
        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            # plot_pid_score_grouped_by_model(res, exp)
            kruskal_rewards(exp, res)
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                     "low_variance_low_cost"]:
            # plot_pid_clicks_grouped_by_model(res, exp)
            # linear_regression_clicks(res, exp)
            kruskal_clicks(exp, res)

    # result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "strategy_discovery_discovered_pid")
    # model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
    # res = group_pid_by_bic(result_df)
    # plot_pid_score_grouped_by_model(res)
