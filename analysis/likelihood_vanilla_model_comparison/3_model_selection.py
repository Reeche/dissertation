import pandas as pd
import numpy as np
import os
from vars import learning_participants
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

"""
Compare vanilla models based on the fit
"""


def compare_pseudo_likelihood(data, trials):
    BIC = 2 * data["click_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


def compare_number_of_clicks_likelihood(data, trials):
    BIC = 2 * data["mer_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


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
    data = data[data["model"].isin(["1743", "1756", "479", "491", "522", "full"])]
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    data = data.sort_index()  # 1743, 1756, 479, 491, 522, mb
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

    ## optional filter only for 1743, 491, 479
    data = data[data["model"].isin(["1743", "491", "479"])]

    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]
    return res


def plot_pid_score_grouped_by_model(data, exp=None):
    # plot the score of the participants who are best explained by a certain model

    model_dict = {
        "Reinforce": "491",
        "LVOC": "479",
        "Habitual": "1743",
        "Not learning": "1756",
        "SSL": "522",
        "Model-based": "full"}

    for model_type, models in model_dict.items():
        # filter for the model in data
        filtered_data = data[data["model"] == models]
        # print(len(filtered_data), "unique pid are best explained by the model", model_type)

        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(
            lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)


        # plot the average
        plt.plot(pid_rewards_average, label=f"{model_type}, N={len(filtered_data)}")
        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend()

    # save the plot
    # plt.show()

    # if no dir, create dir

    if not os.path.exists(f"plot/{exp}"):
        os.makedirs(f"plot/{exp}")

    plt.savefig(f"plot/{exp}/pid_grouped_by_model.png")
    plt.close()

    return None


def anova(data):
    # anova for the last 60 trials across all models

    # get only relevant columns model, pid_rewards
    data = data[["model", "pid_rewards"]]

    # convert pid_rewards to list
    data['pid_rewards'] = data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

    # for each pid_rewards, calculate the average of the last 60 trials
    data["pid_rewards"] = data["pid_rewards"].apply(lambda x: np.mean(x[-60:]))

    # groupby model and conduct anova using statsmodel
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    #
    # model = ols('pid_rewards ~ model', data=data).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)

    ## Kruskal-Wallis H-test
    from scipy.stats import kruskal
    x = data[data["model"] == "1743"]["pid_rewards"]
    y = data[data["model"] == "491"]["pid_rewards"]
    z = data[data["model"] == "479"]["pid_rewards"]

    res = kruskal(x, y, z)
    print(res)

    return None


def mann_whitney_u_test(data, models):
    # model 1:
    data_a = data[data["model"] == models[0]]
    data_a['pid_rewards'] = data_a['pid_rewards'].apply(
        lambda s: [int(num) for num in s.strip('[]').split()])
    pid_rewards_a = np.array(data_a["pid_rewards"].to_list())
    pid_rewards_average_a = np.mean(pid_rewards_a, axis=0)
    # last 60 trials
    pid_rewards_average_a = pid_rewards_average_a[-60:]

    # model 2:
    data_b = data[data["model"] == models[1]]
    data_b['pid_rewards'] = data_b['pid_rewards'].apply(
        lambda s: [int(num) for num in s.strip('[]').split()])
    pid_rewards_b = np.array(data_b["pid_rewards"].to_list())
    pid_rewards_average_b = np.mean(pid_rewards_b, axis=0)
    # last 60 trials
    pid_rewards_average_b = pid_rewards_average_b[-60:]

    res = mannwhitneyu(pid_rewards_average_b, pid_rewards_average_a, alternative="greater")
    print(f"Mann whitney U for models {models[0]} and model {models[1]}: {res}")
    print(f"Mean average score of model {models[0]}: {np.mean(pid_rewards_average_a[-60:])}")
    print(f"Mean average score of model {models[1]}: {np.mean(pid_rewards_average_b[-60:])}")
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

        # add BIC
        # if exp in ["v1.0", "c1.1", "c2.1"]:
        #     data["BIC"] = compare_pseudo_likelihood(data, 35)
        # elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]:
        #     data["BIC"] = compare_number_of_clicks_likelihood(data, 35)
        # elif exp == "strategy_discovery":
        #     data["BIC"] = compare_pseudo_likelihood(data, 120)

        if exp == "strategy_discovery":
            data["BIC"] = compare_loss(data, 120)
        else:
            data["BIC"] = compare_loss(data, 35)

        data = data[data["pid"].isin(learning_participants[exp])]
        df_all.append(data)

        result_df = pd.concat(df_all, ignore_index=True)
        res = group_pid_by_bic(result_df)
        plot_pid_score_grouped_by_model(res, exp)
        # mann_whitney_u_test(res, ["1743", "479"])
        # anova(res)

    # result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "test")
    # model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
    # res = group_pid_by_bic(result_df)
    # plot_pid_score_grouped_by_model(res)
