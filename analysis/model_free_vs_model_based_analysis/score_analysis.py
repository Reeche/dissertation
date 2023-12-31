import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from vars import clicked_dict

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


def plot_expected_score(conditions, clicked_dict):
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))

        selected_columns = clicked_dict[condition]
        selected_columns = [col for col in selected_columns if
                            col in data.columns]  # convert to the same type as header
        selected_df = data[selected_columns]

        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_df = selected_df - 1
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
    # plt.show()
    plt.savefig(f"plots/expected_score.png")
    plt.close()
    return None


def compare_expected_score(conditions, clicked_dict):
    ### statistical test
    # create a dataframe with score, trial and condition
    exp_score_data = pd.DataFrame()
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))
        # select only columns in data that are in clicked_dict
        selected_columns = clicked_dict[condition]
        selected_columns = [col for col in selected_columns if
                            col in data.columns]  # convert to the same type as header
        selected_df = data[selected_columns]

        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_df = selected_df - 1
        # replace strategy with score
        strategy_df = strategy_df.replace(score_mapping)

        reshaped_df = pd.melt(strategy_df.reset_index(), id_vars='index', var_name='trial', value_name='score')
        reshaped_df["condition"] = condition

        # if condition is MF, remove the first 15 trials from the dataframe
        if condition == "mf":
            reshaped_df = reshaped_df[reshaped_df["index"] >= 14]
            reshaped_df["index"] = reshaped_df["index"] - 14

        # create a dataframe with score, trial and condition
        exp_score_data = exp_score_data._append(reshaped_df)
    exp_score_data.columns = ["trial", "pid", "score", "condition"]

    ## linear regression
    regression_analysis(exp_score_data)

    ## Mann Whitney U test
    mann_whitney_test(exp_score_data)


def compare_actual_score(conditions, clicked_dict):
    ### statistical test
    # create a dataframe with score, trial and condition
    score_data = pd.DataFrame()
    for condition in conditions:
        data = pd.read_csv(f"../../data/human/{condition}/mouselab-mdp.csv")
        # filter column "pid" for mf_clicked
        data = data[data["pid"].isin(clicked_dict[condition])]

        # reset trial index to start with 1
        data["trial_index"] = data.groupby("pid").cumcount() + 1

        # if condition is MF, remove the first 15 trials from the dataframe and reset index
        if condition == "mf":
            data = data[data["trial_index"] >= 14]
            data["trial_index"] = data["trial_index"] - 14

        # keep only the columns "score", "trial_index" and "condition"
        data = data[["score", "trial_index", "condition"]]

        # set column condition = condition
        data["condition"] = condition

        # create a dataframe with score, trial and condition
        score_data = score_data._append(data)
    score_data.columns = ["score", "trial", "condition"]

    ## linear regression
    regression_analysis(score_data)

    ## Mann Whitney U test
    mann_whitney_test(score_data)


def regression_analysis(exp_score_data):
    res = ols('score ~ trial*C(condition, Treatment("mf"))', data=exp_score_data).fit()
    print(res.summary())
    return None


def mann_whitney_test(score_data):
    ## Mann Whitney U test

    # filter for first trial
    score_data = score_data[score_data["trial"] == 14]

    # filter data for mf
    mf_exp_score = score_data[score_data["condition"] == "mf"]

    # filter data for mb
    mb_exp_score = score_data[score_data["condition"] == "mb"]

    # filter data for stroop
    stroop_exp_score = score_data[score_data["condition"] == "stroop"]

    # print the mean of expected score for each condition
    print("Mean of expected score for each condition")
    print("MF: ", mf_exp_score["score"].mean())
    print("MB: ", mb_exp_score["score"].mean())
    print("STROOP: ", stroop_exp_score["score"].mean())

    # sd within each group
    print("SD of expected score for each condition")
    print("MF: ", mf_exp_score["score"].std())
    print("MB: ", mb_exp_score["score"].std())
    print("STROOP: ", stroop_exp_score["score"].std())

    # kruskal wallis test
    print("Kruskal Wallis test for expected score between MF, MB and STROOP")
    print(stats.kruskal(mf_exp_score["score"], mb_exp_score["score"], stroop_exp_score["score"]))

    # compare mf and mb
    print("Mann Whitney U test for expected score between MF and MB")
    print(stats.mannwhitneyu(mf_exp_score["score"], mb_exp_score["score"], alternative="greater"))
    # compare mf and stroop
    print("Mann Whitney U test for expected score between MF and STROOP")
    print(stats.mannwhitneyu(mf_exp_score["score"], stroop_exp_score["score"], alternative="greater"))
    # compare mb and stroop
    print("Mann Whitney U test for expected score between MB and STROOP")
    print(stats.mannwhitneyu(mb_exp_score["score"], stroop_exp_score["score"], alternative="two-sided"))

    return None


if __name__ == "__main__":
    adaptive = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84, 45, 11, 6, 7,
                29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    # maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    # others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    # plot_expected_score(conditions, clicked_dict)
    compare_expected_score(conditions, clicked_dict)

    # compare_actual_score(conditions, clicked_dict)
