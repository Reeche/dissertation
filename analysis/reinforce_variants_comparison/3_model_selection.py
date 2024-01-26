import pandas as pd
import numpy as np
import pymannkendall as mk
from vars import learning_participants, clicking_participants, model_dict
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


def create_csv_for_matlab(data, exp):
    # create csv for matlab; filter for required models
    data = data[data["model"].isin([480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491])]
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    data = data.sort_index()
    # data = missing_bic(data)
    data.to_csv(f"matlab/{exp}.csv", index=False, header=False)


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
    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]
    return res


def plot_pid_score_grouped_by_model(exp, data):
    # plot the score of the participants who are best explained by a certain model
    for model_type, models in model_dict.items():
        # filter for the model in data
        filtered_data = data[data["model"].isin(models)]
        print(len(filtered_data), "unique pid are best explained by the the model", model_type)

        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(
            lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)

        # print the average reward of the last 60 trials
        # print("Average reward of the last 60 trials of model:", model_type, np.mean(pid_rewards_average[-60:]))

        # plot the average
        plt.plot(pid_rewards_average, label=f"{model_type}, N={len(filtered_data)}")
        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend()

    # save the plot
    # plt.show()
    plt.savefig(f"plots/{exp}.png")
    plt.close()

    return None


def analyse_subjective_cost(exp, res):
    df = pd.read_csv(f"parameters/{exp}_parameters.csv", index_col=0)

    # filter for models with sc
    # df = df[df["model"].isin(model_dict["SC"])]

    # filter for models in the model_dict for sc
    filtered_pid = res[res["model"].isin(model_dict["SC"])]
    merged_df = pd.merge(df, filtered_pid, on=['pid', 'model'], how='inner')
    df = merged_df

    df["parameters"] = df["parameters"].apply(lambda x: eval(x))

    # iterate through all rows and get the subjective cost
    for index, row in df.iterrows():
        df.loc[index, "subjective_cost"] = row["parameters"]["subjective_cost"]

    # df["subjective_cost"] = df["parameters"][0]["subjective_cost"]

    # plot a histogram of the subjective cost
    # plt.hist(merged_df["subjective_cost"])
    # plt.show()
    print(exp)
    print("subjective_cost mean", df["subjective_cost"].mean())
    # print("subjective_cost std", df["subjective_cost"].std())
    print("subjective_cost upper", df["subjective_cost"].mean() + 1.96 * df["subjective_cost"].std() / np.sqrt(len(df)))
    print("subjective_cost lower", df["subjective_cost"].mean() - 1.96 * df["subjective_cost"].std() / np.sqrt(len(df)))
    return None


def compare_parameters_adaptive(exp, data):
    """
    Compare the learning rate and exploration rate between very adaptive and other participants
    adaptiveness is measured by score (v1.0, c2.1, c1.1) or number of clicks (HVHC, HVLC, LVHC, LVLC)

    Args:
        exp:
        data:

    Returns:

    """

    # keep only data for 491 model
    data = data[data["model"] == 491]

    # classify participants into adaptive and other
    data['pid_rewards'] = data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

    if exp in ["v1.0", "c1.1", "c2.1"]:
        very_adaptive_pid = []
        # Mann kendall test on pid_rewards and append significant result pid into a list
        for pid in data["pid"].unique():
            pid_rewards = data[data["pid"] == pid]["pid_rewards"].tolist()[0]
            result = mk.original_test(pid_rewards)
            if result.p < 0.05:
                very_adaptive_pid.append(pid)
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        very_adaptive_pid = []
        # get number of clicks by counting len of pid_clicks
        for pid in data["pid"].unique():
            pid_data = eval(data[data["pid"] == pid]["pid_clicks"].iloc[0])
            num_clicks = [len(sub_list) for sub_list in pid_data]
            result = mk.original_test(num_clicks)
            if result.p < 0.05:
                very_adaptive_pid.append(pid)

    # load parameters data
    df = pd.read_csv(f"parameters/{exp}_parameters.csv", index_col=0)
    df["parameters"] = df["parameters"].apply(lambda x: eval(x))

    # filter for model 491
    df = df[df["model"] == 491]

    # filter for adaptive participants
    df_adaptive = df[df["pid"].isin(very_adaptive_pid)]

    # get the parameters learning_rate and inverse_temperature
    for index, row in df_adaptive.iterrows():
        df_adaptive.loc[index, "lr"] = row["parameters"]["lr"]
        df_adaptive.loc[index, "inverse_temperature"] = row["parameters"]["inverse_temperature"]

    # filter for other participants
    df_other = df[~df["pid"].isin(very_adaptive_pid)]

    # get the parameters learning_rate and inverse_temperature
    for index, row in df_other.iterrows():
        df_other.loc[index, "lr"] = row["parameters"]["lr"]
        df_other.loc[index, "inverse_temperature"] = row["parameters"]["inverse_temperature"]

    # Mann Whitney U test whether lr and inverse_temperature are different between adaptive and other participants
    print(exp)
    print("learning rate", mannwhitneyu(df_adaptive["lr"], df_other["lr"], method="exact"))
    print("inverse_temperature",
          mannwhitneyu(df_adaptive["inverse_temperature"], df_other["inverse_temperature"], method="exact"))

    # mean and std of lr and inverse_temperature
    print("lr adaptive", df_adaptive["lr"].mean(), df_adaptive["lr"].std())
    print("lr other", df_other["lr"].mean(), df_other["lr"].std())
    print("inverse_temperature adaptive", df_adaptive["inverse_temperature"].mean(),
          df_adaptive["inverse_temperature"].std())
    print("inverse_temperature other", df_other["inverse_temperature"].mean(),
          df_other["inverse_temperature"].std())

    return None


if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    # experiment = ["high_variance_high_cost"]
    experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
    # experiment = ["strategy_discovery"]
    df_all = []
    for exp in experiment:

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

        ## conduct parameter analysis for adaptive and other participants without the learning/clicking filter and for vanilla REINFORCE or RL + PR model
        compare_parameters_adaptive(exp, data)

        # if exp in ["v1.0", "c1.1", "c2.1"]:
        #     data = data[data["pid"].isin(clicking_participants[exp])]
        # elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
        #              "low_variance_low_cost"]:
        #     data = data[data["pid"].isin(learning_participants[exp])]

        ## for each condition analyse the parameters
        # res = group_pid_by_bic(data)
        # analyse_subjective_cost(exp, res)

        # df_all.append(data)

    # result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "lvlc")
    # model_bic = sort_by_BIC(result_df)

    # res = group_pid_by_bic(result_df)
    # plot_pid_score_grouped_by_model("lvlc", res)
    # analyse_subjective_cost(res)
