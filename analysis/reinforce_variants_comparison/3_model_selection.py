import pandas as pd
import numpy as np
from vars import learning_participants
import matplotlib.pyplot as plt

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
    data = data[data["model"].isin([480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491])]
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    data = data.sort_index()
    # data = missing_bic(data)
    data.to_csv(f"matlab/{exp}.csv", index=False, header=False)

def missing_bic(df):
    #replace the missing value by row and column average
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

def plot_pid_score_grouped_by_model(data):
    # plot the score of the participants who are best explained by a certain model

    model_dict = {
    "PR": [480, 481, 482, 483, 484, 485, 486, 487],
    "TD": [480, 482, 484, 486, 488, 490],
    "SC": [480, 481, 484, 485, 488, 489],
    "PR + TD": [480, 482, 484, 486, 488],
    "Vanilla": [491],
    # "LVOC": [479],
    "Habitual": [1743]}

    # todo: how many are best explained by the overlap of PR and TD

    # model_dict = {
    # "PR": [483, 487],
    # "TD": [490],
    # "SC": [489]}

    for model_type, models in model_dict.items():
        # filter for the model in data
        filtered_data = data[data["model"].isin(models)]
        print(len(filtered_data), "unique pid are best explained by the the model", model_type)

        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])

        # calculate the average of the pid_rewards
        pid_rewards = np.array(filtered_data["pid_rewards"].to_list())
        pid_rewards_average = np.mean(pid_rewards, axis=0)

        # print the average reward of the last 60 trials
        print("Average reward of the last 60 trials of model:", model_type, np.mean(pid_rewards_average[-60:]))

        # plot the average
        plt.plot(pid_rewards_average, label=f"{model_type}, N={len(filtered_data)}")
        plt.xlabel("Trial")
        plt.ylabel("Average score")
        plt.legend()

    # save the plot
    plt.show()
    # plt.savefig(f"plots/{model}.png")
    plt.close()

    return None

if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    experiment = ["strategy_discovery"]
    # experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
    # experiment = ["strategy_discovery"]
    df_all = []
    for exp in experiment:

        data = pd.read_csv(f"data/{exp}_w_habit.csv", index_col=0)

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
    # create_csv_for_matlab(result_df, "strategy_discovery")
    model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
    res = group_pid_by_bic(result_df)
    plot_pid_score_grouped_by_model(res)

