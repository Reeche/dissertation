import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""
Analyse whether score and expected score differs across the conditions
"""


def compare_score(conditions, mf_clicked):
    for condition in conditions:
        # compare and plot and score of all the conditions: MF, MB, STROOP
        df = pd.read_csv(f"../../data/human/{condition}/mouselab-mdp.csv")
        # filter for data where "block" is "training"
        df = df[df["block"] == "training"]
        df = df[["pid", "trial_index", "score"]]

        if condition == "mf":
            df = df[df["pid"].isin(mf_clicked)]

        # reset trial_index; each pid has trial 1-30
        df["trial_index"] = df.groupby("pid").cumcount() + 1

        # average score for all participants across trials
        df_score = df.groupby(["trial_index"]).mean()

        if condition == "mf":
            x_values = list(range(30))
            plt.plot(x_values, df_score["score"], label=condition)
        else:
            x_values = list(range(15, 30))
            plt.plot(x_values, df_score["score"], label=condition)
    return None


def plot_expected_score(conditions, mf_clicked):
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))

        # filter data for mf_clicked
        if condition == "mf":
            data = data[mf_clicked]

        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_df = data - 1
        # replace strategy with score
        strategy_df = strategy_df.replace(score_mapping)

        # calculate average score for each trial
        average = strategy_df.mean(axis=1)

        # plot the average
        if condition == "mf":
            x_values = list(range(30))
            plt.plot(x_values, average, label=condition.upper())
        else:
            x_values = list(range(15, 30))
            plt.plot(x_values, average, label=condition.upper())

        # add 95% confidence interval
        # calculate standard error
        standard_error = strategy_df.sem(axis=1)
        # calculate upper and lower bound
        upper_bound = average + 1.96 * standard_error
        lower_bound = average - 1.96 * standard_error
        # plot the upper and lower bound
        plt.fill_between(x_values, upper_bound, lower_bound, alpha=0.2)

    plt.xlabel("Trial")
    plt.ylabel("Expected score")
    plt.legend()
    plt.show()
    plt.close()
    return None


def compare_expected_score(conditions, mf_clicked):
    ### statistical test
    # create a dataframe with score, trial and condition
    exp_score_data = pd.DataFrame(columns=["exp_score", "trial", "condition"])
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))
        # filter data for mf_clicked
        if condition == "mf":
            data = data[mf_clicked]

        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_df = data - 1
        # replace strategy with score
        strategy_df = strategy_df.replace(score_mapping)

        reshaped_df = pd.melt(strategy_df.reset_index(), id_vars='index', var_name='trial', value_name='exp_score')
        reshaped_df["condition"] = condition

        # if condition is MF, remove the first 15 trials from the dataframe
        if condition == "mf":
            reshaped_df = reshaped_df[reshaped_df["index"] >= 15]
            reshaped_df["index"] = reshaped_df["index"] - 15

        # create a dataframe with score, trial and condition
        exp_score_data = exp_score_data._append(reshaped_df)
    exp_score_data.columns = ["exp_score", "pid", "condition", "trial"]

    ## linear regression
    regression_analysis(exp_score_data)

    ## Mann Whitney U test
    mann_whitney_test(exp_score_data)


def regression_analysis(exp_score_data):
    res = ols('exp_score ~ trial*C(condition, Treatment("mf"))', data=exp_score_data).fit()
    print(res.summary())
    return None

def mann_whitney_test(exp_score_data):
    ## Mann Whitney U test
    # filter data for mf_clicked
    mf_exp_score = exp_score_data[exp_score_data["condition"] == "mf"]
    # filter for trial == 15
    mf_exp_score = mf_exp_score[mf_exp_score["trial"] == 0]
    # filter data for mb
    mb_exp_score = exp_score_data[exp_score_data["condition"] == "mb"]
    # filter for trial == 15
    mb_exp_score = mb_exp_score[mb_exp_score["trial"] == 0]
    # filter data for stroop
    stroop_exp_score = exp_score_data[exp_score_data["condition"] == "stroop"]
    # filter for trial == 15
    stroop_exp_score = stroop_exp_score[stroop_exp_score["trial"] == 0]

    # print the mean of expected score for each condition
    print("Mean of expected score for each condition")
    print("MF: ", mf_exp_score["exp_score"].mean())
    print("MB: ", mb_exp_score["exp_score"].mean())
    print("STROOP: ", stroop_exp_score["exp_score"].mean())

    # sd within each group
    print("SD of expected score for each condition")
    print("MF: ", mf_exp_score["exp_score"].std())
    print("MB: ", mb_exp_score["exp_score"].std())
    print("STROOP: ", stroop_exp_score["exp_score"].std())


    # compare mf and mb
    print("Mann Whitney U test for expected score between MF and MB")
    print(stats.mannwhitneyu(mf_exp_score["exp_score"], mb_exp_score["exp_score"], alternative="greater"))
    # compare mf and stroop
    print("Mann Whitney U test for expected score between MF and STROOP")
    print(stats.mannwhitneyu(mf_exp_score["exp_score"], stroop_exp_score["exp_score"], alternative="greater"))
    # compare mb and stroop
    print("Mann Whitney U test for expected score between MB and STROOP")
    print(stats.mannwhitneyu(mb_exp_score["exp_score"], stroop_exp_score["exp_score"], alternative="two-sided"))

    return None

if __name__ == "__main__":
    adaptive = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84, 45, 11, 6, 7,
                29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    # maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    # others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    mf_clicked = [3, 5, 9, 10, 13, 15, 23, 25, 28, 30, 32, 33, 36, 37, 41, 45, 46, 52, 56, 58, 59, 62, 63, 66, 68,
                  69, 72, 74, 76, 78, 82, 84, 86, 89, 91, 93, 94, 96, 98, 100, 102, 104, 108, 111, 115, 116, 124,
                  125, 126, 127, 129, 130, 132, 134, 137, 138, 139, 141, 145, 146, 148, 149, 152, 156, 158, 159,
                  163, 167, 168, 172, 173, 175, 176, 179, 180, 182, 184, 186, 187, 189, 190, 191]

    # plot_expected_score(conditions, mf_clicked)
    compare_expected_score(conditions, mf_clicked)
