import pandas as pd
import numpy as np
import ast
import pymannkendall as mk
from vars import learning_participants, clicking_participants, model_dict, model_grouped, model_names, discovery_hybrid
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
import warnings

warnings.filterwarnings("ignore")

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
    data = data.pivot(index="model", columns="pid", values="BIC").T
    if exp != "strategy_discovery":
        # add 491 model from data_old; load the csv from data_old
        data_old = pd.read_csv(f"../likelihood_vanilla_model_comparison/matlab/{exp}.csv", header=None)
        ##get the last column from data_old
        vanilla_model = list(data_old.iloc[:,-2])

        ##append vanilla_model as the last column
        data["Vanilla"] = vanilla_model

    data.columns = ['PR + SC + TD', 'PR + SC', 'PR + TD', 'PR', 'SC + TD', 'SC', 'TD', 'Vanilla']
    data.to_csv(f"matlab/{exp}_hybrid.csv", index=False, header=False)


def group_pid_by_bic(data):
    # which model explains which participant best
    # for each pid, find the model with the lowest BIC
    min_bic_idx = data.groupby('pid')['BIC'].idxmin()
    res = data.loc[min_bic_idx]
    return res


def plot_pid_grouped_by_model(exp, data, criteria):
    # plot the score of the participants who are best explained by a certain model
    for model_type, models in model_grouped.items():
        # filter for the model in data
        filtered_data = data[data["model"].isin(models)]
        print(len(filtered_data), "unique pid are best explained by the the model", model_type)
        print(filtered_data["pid"].unique())
        if criteria == "pid_mer":
            filtered_data[criteria] = filtered_data[criteria].apply(
                lambda x: [int(float(i)) for i in ast.literal_eval(x)])
        elif criteria == "pid_rewards":
            filtered_data[criteria] = filtered_data[criteria].apply(
                lambda s: [int(num) for num in s.strip('[]').split()])
        else:
            filtered_data[criteria] = filtered_data[criteria].apply(lambda x: ast.literal_eval(x))
            lengths_model = []
            lengths_pid = []
            # Iterate through the DataFrame
            for index, row in filtered_data.iterrows():
                lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
                lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
            filtered_data["model_clicks"] = lengths_model
            filtered_data["pid_clicks"] = lengths_pid

        # calculate the average of the pid_rewards
        pid = np.array(filtered_data[criteria].to_list())
        pid_average = np.mean(pid, axis=0)

        # print the average reward of the last 60 trials
        # print("Average reward of the last 60 trials of model:", model_type, np.mean(pid_rewards_average[-60:]))

        # plot the average
        plt.plot(pid_average, label=f"{model_type}, N={len(filtered_data)}")
        if criteria != "pid_clicks":
            if exp == "v1.0":
                plt.ylim(-5, 70)
            elif exp == "c2.1":
                plt.ylim(-5, 70)
            elif exp == "c1.1":
                plt.ylim(0, 25)
        else:
            plt.ylim(0, 12)
        plt.xlabel("Trial", fontsize=12)
        if criteria == "pid_clicks":
            plt.ylabel("Average clicks", fontsize=12)
        elif criteria == "pid_mer":
            plt.ylabel("Average expected reward", fontsize=12)
        elif criteria == "pid_rewards":
            plt.ylabel("Average score", fontsize=12)
        plt.legend(fontsize=12, ncol=2, loc='lower right')

    # save the plot
    plt.savefig(f"plots/{exp}_grouped_{criteria}.png")
    plt.show()
    plt.close()

    return None

def statistical_test(exp, data, criteria):
    # get only relevant columns model, pid_rewards
    data = data[["model", criteria]]

    # convert pid_rewards from strings to list
    if criteria == "pid_mer":
        data[criteria] = data[criteria].apply(lambda x: [int(float(i)) for i in ast.literal_eval(x)])
        # data[criteria] = data[criteria].apply(lambda s: [int(num) for num in s.strip('[]').split()])
    elif criteria == "pid_rewards":
        data[criteria] = data[criteria].apply(lambda s: [int(num) for num in s.strip('[]').split()])
        # data[criteria] = data[criteria].apply(lambda s: [int(num) for num in s.strip('[]').split()])
    elif criteria == "pid_clicks":
        data[criteria] = data[criteria].apply(lambda x: ast.literal_eval(x))
        pid_clicks = np.array(data[criteria].to_list())
        result_array = np.array([[len(cell) - 1 for cell in row] for row in pid_clicks])
        # make this array into a list
        data[criteria] = result_array.tolist()


    if exp == "strategy_discovery":
        data[criteria] = data[criteria].apply(lambda x: x[-60:])
    else:
        data[criteria] = data[criteria].apply(lambda x: x[-10:])

    # get all the mer values of the model 3326 and 3327 together as one list
    model_pr = (data[data["model"] == 3315][criteria].to_list()
                + data[data["model"] == 3316][criteria].to_list()
                + data[data["model"] == 3317][criteria].to_list()
                + data[data["model"] == 3318][criteria].to_list())

    model_no_pr = (data[data["model"] == 3323][criteria].to_list()
                   + data[data["model"] == 3324][criteria].to_list()
                   + data[data["model"] == 3325][criteria].to_list()
                   + data[data["model"] == 3326][criteria].to_list())

    model_td = (data[data["model"] == 3315][criteria].to_list()
                + data[data["model"] == 3323][criteria].to_list()
                + data[data["model"] == 3317][criteria].to_list()
                + data[data["model"] == 3325][criteria].to_list())

    model_no_td = (data[data["model"] == 3316][criteria].to_list()
                   + data[data["model"] == 3318][criteria].to_list()
                   + data[data["model"] == 3324][criteria].to_list()
                   + data[data["model"] == 3326][criteria].to_list())

    model_sc = (data[data["model"] == 3315][criteria].to_list()
                + data[data["model"] == 3316][criteria].to_list()
                + data[data["model"] == 3323][criteria].to_list()
                + data[data["model"] == 3324][criteria].to_list())

    model_no_sc = (data[data["model"] == 3317][criteria].to_list()
                   + data[data["model"] == 3318][criteria].to_list()
                   + data[data["model"] == 3325][criteria].to_list()
                   + data[data["model"] == 3326][criteria].to_list())

    # count how many participants are in each group
    print("PR", len(model_pr))
    print("No PR", len(model_no_pr))
    print("TD", len(model_td))
    print("No TD", len(model_no_td))
    print("SC", len(model_sc))
    print("No SC", len(model_no_sc))

    # print mean and std of each
    print(f"PR mean: {np.mean([item for sublist in model_pr for item in sublist])}")
    print(f"PR std: {np.std([item for sublist in model_pr for item in sublist])}")
    print(f"No PR mean: {np.mean([item for sublist in model_no_pr for item in sublist])}")
    print(f"No PR std: {np.std([item for sublist in model_no_pr for item in sublist])}")

    print(f"TD mean: {np.mean([item for sublist in model_td for item in sublist])}")
    print(f"TD std: {np.std([item for sublist in model_td for item in sublist])}")
    print(f"No TD mean: {np.mean([item for sublist in model_no_td for item in sublist])}")
    print(f"No TD std: {np.std([item for sublist in model_no_td for item in sublist])}")

    print(f"SC mean: {np.mean([item for sublist in model_sc for item in sublist])}")
    print(f"SC std: {np.std([item for sublist in model_sc for item in sublist])}")
    print(f"No SC mean: {np.mean([item for sublist in model_no_sc for item in sublist])}")
    print(f"No SC std: {np.std([item for sublist in model_no_sc for item in sublist])}")

    if not np.isnan(np.mean([item for sublist in model_pr for item in sublist])):
        pr = mannwhitneyu([item for sublist in model_pr for item in sublist],
                          [item for sublist in model_no_pr for item in sublist], alternative="two-sided")
        print(f"Mann Whitney U PR vs no PR: ", pr)

    if not np.isnan(np.mean([item for sublist in model_td for item in sublist])):
        td = mannwhitneyu([item for sublist in model_td for item in sublist],
                          [item for sublist in model_no_td for item in sublist], alternative="two-sided")
        print(f"Mann Whitney U TD vs no TD: ", td)

    if not np.isnan(np.mean([item for sublist in model_sc for item in sublist])):
        sc = mannwhitneyu([item for sublist in model_sc for item in sublist],
                          [item for sublist in model_no_sc for item in sublist], alternative="two-sided")
        print(f"Mann Whitney U SC vs no SC: ", sc)

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

def analyse_subjective_cost(res, exp=None):
    if exp:
        df = pd.read_csv(f"parameters/{exp}_parameters.csv", index_col=0)
    else:
        df1 = pd.read_csv(f"parameters/high_variance_high_cost_parameters.csv", index_col=0)
        df2 = pd.read_csv(f"parameters/high_variance_low_cost_parameters.csv", index_col=0)
        df3 = pd.read_csv(f"parameters/low_variance_high_cost_parameters.csv", index_col=0)
        df4 = pd.read_csv(f"parameters/low_variance_low_cost_parameters.csv", index_col=0)
        # concatenate all the dataframes and keep information about the experiment
        df = pd.concat([df1, df2, df3, df4], ignore_index=True)


    # filter for models in the model_dict for sc
    filtered_pid = res[res["model"].isin(model_dict["SC"])]
    merged_df = pd.merge(df, filtered_pid, on=['pid', 'model'], how='inner')
    df = merged_df

    df["parameters"] = df["parameters"].apply(lambda x: eval(x))

    # iterate through all rows and get the subjective cost
    for index, row in df.iterrows():
        df.loc[index, "subjective_cost"] = row["parameters"]["subjective_cost"]


    # create one boxplot for each experiment all in one plot and add mean
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[df["exp_x"] == "high_variance_high_cost"]["subjective_cost"],
                    df[df["exp_x"] == "high_variance_low_cost"]["subjective_cost"],
                    df[df["exp_x"] == "low_variance_high_cost"]["subjective_cost"],
                    df[df["exp_x"] == "low_variance_low_cost"]["subjective_cost"]], showmeans=True)
    plt.xticks([1, 2, 3, 4], ["HVHC", "HVLC", "LVHC", "LVLC"], fontsize=14)
    # add the points as data next to each of the box
    for i, exp in enumerate(["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]):
        plt.scatter([i+1]*len(df[df["exp_x"] == exp]), df[df["exp_x"] == exp]["subjective_cost"], alpha=0.3, color="black")
    plt.ylabel("Subjective cost", fontsize=14)
    # plt.show()
    plt.savefig("plots/subjective_cost.png")
    plt.close()

    # for each exp print mean and std of subjective cost
    print("HVHC", df[df["exp_x"] == "high_variance_high_cost"]["subjective_cost"].mean(),
          df[df["exp_x"] == "high_variance_high_cost"]["subjective_cost"].std())
    print("HVLC", df[df["exp_x"] == "high_variance_low_cost"]["subjective_cost"].mean(),
          df[df["exp_x"] == "high_variance_low_cost"]["subjective_cost"].std())
    print("LVHC", df[df["exp_x"] == "low_variance_high_cost"]["subjective_cost"].mean(),
          df[df["exp_x"] == "low_variance_high_cost"]["subjective_cost"].std())
    print("LVLC", df[df["exp_x"] == "low_variance_low_cost"]["subjective_cost"].mean(),
          df[df["exp_x"] == "low_variance_low_cost"]["subjective_cost"].std())

    # test whether the subjective cost is significantly larger than 0 using wilcoxon
    for exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]:
        sample_data = df[df["exp_x"] == exp]["subjective_cost"]
        print(exp, wilcoxon(sample_data, alternative="greater"))
    return None


def list_pid_lowest_bic(res):
    # for each unique model, get list of pid
    for model in res["model"].unique():
        print(model_names[model])
        print(res[res["model"] == model]["pid"].unique())
        print("Number of participants:", len(res[res["model"] == model]["pid"].unique()))
    return None

if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    # experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
    experiment = ["strategy_discovery"]
    df_all = []
    for exp in experiment:
        print(exp)

        data = pd.read_csv(f"data/{exp}.csv", index_col=0)

        ## add vanilla model; only relevant for SD?
        if exp == "strategy_discovery":
            vanilla_data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv")
            vanilla_data = vanilla_data[vanilla_data["model_index"] == "3326"]
            vanilla_data["model"] = 3326
            vanilla_data["exp"] = "strategy_discovery"
            vanilla_data = vanilla_data.drop(columns=["class", "model_index", "condition", "Unnamed: 0.1", "Unnamed: 0"])
            data = pd.concat([data, vanilla_data], ignore_index=True)

        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            # data = data[data["pid"].isin(clicking_participants[exp])]
            data = data[data["pid"].isin(discovery_hybrid)]
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                     "low_variance_low_cost"]:
            data = data[data["pid"].isin(learning_participants[exp])]

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
        # compare_parameters_adaptive(exp, data)

        df_all.append(data)

        result_df = pd.concat(df_all, ignore_index=True)
        # create_csv_for_matlab(result_df, exp)

        res = group_pid_by_bic(result_df)

        # get the list of pid who are best explained by a variant
        # list_pid_lowest_bic(res)

        # plot_pid_grouped_by_model(exp, res, "pid_rewards")
        statistical_test(exp, res, "pid_clicks")
        # analyse_subjective_cost(exp, res)
        # parameters_analysis(res, exp)

    # analyse_subjective_cost(res)
    # model_bic = sort_by_BIC(result_df)
