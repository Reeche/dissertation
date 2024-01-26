import pandas as pd
import numpy as np
import os
from vars import clicking_participants, learning_participants, discovered_participants
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import ast

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
    data = data[data["model"].isin([1743, 1756, 479, 491, 522])]
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    # data = data.sort_index()  # 1743, 1756, 479, 491, 522, mb
    ## for strategy discovery the order is: 479, 491, 522, 1743, 1756
    # data = missing_bic(data)
    data.to_csv(f"matlab/{exp}.csv", index=False, header=False)


def remove_double_mb_entries(data):
    # remove entries from the data where the model is mb and the number of parameters is 4
    data = data[~((data["model"] == "mb") & (data["number_of_parameters"] == 4))]
    # remove duplicates
    data = data.drop_duplicates(subset=["pid", "model"])
    # save as csv
    # data.to_csv(f"{exp}_{criterion}.csv", index=False)
    return data


def missing_bic(df):
    # replace the missing value by row and column average
    # Calculate row averages
    row_avg = df.mean(axis=1, skipna=True).tolist()

    # Calculate column averages
    col_avg = df.mean(axis=0, skipna=True).tolist()

    # Iterate through DataFrame
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if pd.isna(df.iat[i, j]):
                df.iat[i, j] = (row_avg[i] + col_avg[j]) / 2

    return df


def group_pid_by_bic(data):
    # which model explains which participant best

    ## optional filter only for 1743, 491, 479 (1756 is not learning)
    # data = data[data["model"].isin([1743, 1756, 491, 479, 522])]
    data = data[data["model"].isin([1743, 491, 479, 522])]

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


def plot_pid_score_grouped_by_model(data, exp=None):
    # plot the score of the participants who are best explained by a certain model

    model_dict = {
        "Reinforce": 491,
        "LVOC": 479,
        "Habitual": 1743,
        # "Not learning": 1756,
        "SSL": 522}
        # "Model-based": "full"}

    for model_type, models in model_dict.items():
        # filter for the model in data
        filtered_data = data[data["model"] == models]
        print(len(filtered_data), "unique pid are best explained by the model", model_type)

        # filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)

        # plot the average
        plt.plot(pid_rewards_average, label=f"{model_type}, N={len(filtered_data)}")

        # add 95% confidence interval
        plt.fill_between(np.arange(1, 121), pid_rewards_average - 1.96 * np.std(pid_rewards, axis=0),
                         pid_rewards_average + 1.96 * np.std(pid_rewards, axis=0), alpha=0.2)

        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend()

    # save the plot
    # plt.show()

    # if no dir, create dir

    if not os.path.exists(f"plot/{exp}"):
        os.makedirs(f"plot/{exp}")

    plt.savefig(f"plot/{exp}/only_learning_models.png")
    plt.close()

    return None


def kruskal(exp, data):
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
    from scipy.stats import kruskal
    q = data[data["model"] == 1756]["pid_rewards"]
    x = data[data["model"] == 1743]["pid_rewards"]
    y = data[data["model"] == 491]["pid_rewards"]
    z = data[data["model"] == 479]["pid_rewards"]


    res = kruskal(q, x, y, z)
    print(res)

    return None

def mann_whitney_u_test(exp, data, models):
    ## test whether average score of pid of a model pair is significantly different

    # model 1:
    data_a = data[data["model"] == models[0]]
    data_a['pid_rewards'] = data_a['pid_rewards'].apply(
        lambda s: [int(num) for num in s.strip('[]').split()])
    pid_rewards_a = np.array(data_a["pid_rewards"].to_list())
    pid_rewards_average_a = np.mean(pid_rewards_a, axis=0)
    # last 60 trials
    if exp == "strategy_discovery":
        last_trials = 60
    else:
        last_trials = 10
    pid_rewards_average_a = pid_rewards_average_a[-last_trials:]

    # model 2:
    data_b = data[data["model"] == models[1]]
    data_b['pid_rewards'] = data_b['pid_rewards'].apply(
        lambda s: [int(num) for num in s.strip('[]').split()])
    pid_rewards_b = np.array(data_b["pid_rewards"].to_list())
    pid_rewards_average_b = np.mean(pid_rewards_b, axis=0)
    # last 60 trials
    pid_rewards_average_b = pid_rewards_average_b[-last_trials:]

    res = mannwhitneyu(pid_rewards_average_b, pid_rewards_average_a, alternative="greater")
    print(f"Mann whitney U for models {models[0]} and model {models[1]}: {res}")
    print(f"Mean average score of model {models[0]}: {np.mean(pid_rewards_average_a[-last_trials:])}")
    print(f"Mean average score of model {models[1]}: {np.mean(pid_rewards_average_b[-last_trials:])}")
    return None


if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    # experiment = ["v1.0"]
    # experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
    experiment = ["strategy_discovery"]
    # df_all = []
    for exp in experiment:
        df_all = []
        data = pd.read_csv(f"data/{exp}.csv", index_col=0)

        calculate_score_loss(data)

        if exp == "strategy_discovery":
            data["BIC"] = compare_loss(data, 120)
        else:
            data["BIC"] = compare_loss(data, 35)

        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            data = data[data["pid"].isin(clicking_participants[exp])]
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]:
            data = data[data["pid"].isin(learning_participants[exp])]
        df_all.append(data)

        ### for individual analysis
        result_df = pd.concat(df_all, ignore_index=True)
        res = group_pid_by_bic(result_df) # get BIC for only selected models
        # res = group_pid_by_score(result_df)
        plot_pid_score_grouped_by_model(res, exp)
        # kruskal(exp, res)
        # mann_whitney_u_test(exp, res, [1743, 1756])
        # mann_whitney_u_test(exp, res, [1743, 479])
        # mann_whitney_u_test(exp, res, [1743, 491])
        # mann_whitney_u_test(exp, res, [491, 479])


    # result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "strategy_discovery_discovered_pid")
    # model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
    # res = group_pid_by_bic(result_df)
    # plot_pid_score_grouped_by_model(res)
