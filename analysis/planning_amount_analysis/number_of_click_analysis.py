import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from collections import Counter
from scipy.stats import chisquare, chi2_contingency
import os
from vars import clicking_pid, learning_pid

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
stats = importr('stats')

"""
This file contains the new analysis for planning amount experiment for the journal paper submission
Changes compared to NeurIPS submission: 
* Now contains al data, i.e. the participants who clicked nothing throughout all trials are now included

"""


def create_click_df(data, experiment):
    """
    Create a df containing "pid", "trial", "number_of_clicks", "clicks", "click_cost"
    Args:
        data: mouselab-mdp raw data with "queries" containing all crucial information

    Returns: a df containing "pid", "trial", "number_of_clicks", "clicks"

    """
    click_temp_df = data["queries"]
    click_df = pd.DataFrame(columns=["pid", "trial", "variance", "number_of_clicks", "clicks"])

    # create a list of all pid * number of trials and append their clicks
    click_df["pid"] = data["pid"]
    click_df["trial"] = data["trial_index"]

    # get their number of clicks
    number_of_clicks_list = []
    clicks_list = []
    for index, row in click_temp_df.items():
        temp = ast.literal_eval(row)
        clicks = temp["click"]["state"]["target"]
        clicks_list.append(clicks)
        len_of_clicks = len(clicks)
        number_of_clicks_list.append(len_of_clicks)

    click_df["clicks"] = clicks_list
    click_df["number_of_clicks"] = number_of_clicks_list

    if experiment == "high_variance_low_cost":
        click_df["click_cost"] = [0] * len(click_temp_df)
        click_df["variance"] = [1] * len(click_temp_df)
    elif experiment == "high_variance_high_cost":
        click_df["click_cost"] = [1] * len(click_temp_df)
        click_df["variance"] = [1] * len(click_temp_df)
    elif experiment == "low_variance_low_cost":
        click_df["click_cost"] = [0] * len(click_temp_df)
        click_df["variance"] = [0] * len(click_temp_df)
    elif experiment == "low_variance_high_cost":
        click_df["click_cost"] = [1] * len(click_temp_df)
        click_df["variance"] = [0] * len(click_temp_df)

    # if experiment == "high_variance_low_cost":
    #     click_df["click_cost"] = ["low"] * len(click_temp_df)
    #     click_df["variance"] = ["high_variance_low_cost"] * len(click_temp_df)
    # elif experiment == "high_variance_high_cost":
    #     click_df["click_cost"] = ["high"] * len(click_temp_df)
    #     click_df["variance"] = ["high_variance_high_cost"] * len(click_temp_df)
    # elif experiment == "low_variance_low_cost":
    #     click_df["click_cost"] = ["low"] * len(click_temp_df)
    #     click_df["variance"] = ["low_variance_low_cost"] * len(click_temp_df)
    # elif experiment == "low_variance_high_cost":
    #     click_df["click_cost"] = ["high"] * len(click_temp_df)
    #     click_df["variance"] = ["low_variance_high_cost"] * len(click_temp_df)
    click_df["condition"] = experiment
    return click_df


def plot_clicks(average_clicks):
    ci = 1.96 * np.std(average_clicks) / np.sqrt(len(average_clicks))
    plt.plot(average_clicks)
    plt.fill_between(range(len(average_clicks)), average_clicks - ci, average_clicks + ci, color="b",
                     alpha=.1)
    plt.ylim(top=9)
    plt.ylim(bottom=-1)
    plt.xlabel("Trial Number", size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if experiment == "high_variance_low_cost":
        label = "HVLC"
        plt.axhline(y=7.10, color='r', linestyle='-')
    elif experiment == "high_variance_high_cost":
        label = "HVHC"
        plt.axhline(y=6.32, color='r', linestyle='-')
    elif experiment == "low_variance_high_cost":
        label = "LVHC"
        plt.axhline(y=0.01, color='r', linestyle='-')  # it is actually 0 but needs to show on plot, therefore 0.01
    else:
        label = "LVLC"
        plt.axhline(y=5.82, color='r', linestyle='-')
    plt.ylabel(f"Average number of clicks for {label}", fontsize=15)
    plt.savefig(f"plots/{experiment}_average_clicks.png")
    # plt.show()
    plt.close()
    return None


def plot_individual_clicks(click_df, exp):
    pid_list = click_df["pid"].unique()
    divided_list = [pid_list[i:i + 7] for i in range(0, len(pid_list), 7)]
    i = 0
    for lists in divided_list:
        for pid in lists:
            temp_df = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
            plt.plot(temp_df)
        # plt.show()
        plt.savefig(f"plots/{exp}_individual_plots_{i}")
        plt.close()
        i += 1
    return None


def trend_test(average_clicks):
    result = mk.original_test(average_clicks)
    print(f"Mann Kendall test for clicks for {experiment}: clicks are {result}")
    return None


def normality_test(average_clicks):
    k2, p = stats.normaltest(average_clicks)
    print(f"Normality test for clicks for {experiment}: p = {p}")


def anova(click_data):
    model = ols(
        'number_of_clicks ~ C(trial) + click_cost + pid + C(variance) + trial:variance + trial:cost + trial:variance:cost',
        data=click_data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    print(table)
    return None


def lme(click_data):
    # linear mixed effect models
    # formula = "number_of_clicks ~ trial*variance*click_cost"
    # gamma_model = smf.mixedlm(formula=formula, data=click_data, groups=click_data["pid"]).fit()  # makes sense
    # print(gamma_model.summary())

    formula_ = "number_of_clicks ~ trial"
    gamma_model_ = smf.mixedlm(formula=formula_, data=click_data, groups=click_data["pid"]).fit()  # makes sense
    print(gamma_model_.summary())

    return None


def magnitude_of_change(click_df, experiment):
    pid_list = click_df["pid"].unique()
    diff_list = []
    for pid in pid_list:
        temp_list = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
        diff_list.append([t - s for s, t in zip(temp_list, temp_list[1:])])
    flat_list = [item for sublist in diff_list for item in sublist]
    lists = sorted(Counter(flat_list).items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y)
    plt.savefig(f"plots/magnitude_{experiment}")
    plt.close()
    return None


def clicking_pid(click_df, experiment):
    pid_list = click_df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
        if any(v != 0 for v in temp_list):
            good_pid.append(pid)
    print(f"{experiment} number of people who clicked something", len(good_pid), good_pid)
    print("out of ", len(pid_list), "participants")
    return good_pid


def sequential_dependence(data):
    # Fisher test
    click_df = data[['pid', 'trial', 'number_of_clicks']].copy()
    reshaped_click_df = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")
    reshaped_click_df.columns = reshaped_click_df.columns.map(str)

    pairs = []
    for column in reshaped_click_df:
        sequence = reshaped_click_df[column].tolist()
        pairs.append(list(zip(sequence, sequence[1:])))

    all_pairs = [item for sublist in pairs for item in sublist]

    pairs_count_df = pd.DataFrame(0, index=range(0, 13), columns=range(0, 13))
    for pair in all_pairs:
        pairs_count_df[pair[0]][pair[1]] += 1

    pairs_count = pairs_count_df.to_numpy()
    pairs_count_no_diagonal = pairs_count[~np.eye(pairs_count.shape[0], dtype=bool)].reshape(pairs_count.shape[0], -1)
    res = stats.fisher_test(pairs_count_no_diagonal, simulate_p_value=True)
    print(res)


def trend_within_participant(exp, data):
    # how many participants improved significantly
    click_df = data[['pid', 'trial', 'number_of_clicks']].copy()
    reshaped_click_df = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")
    reshaped_click_df.columns = reshaped_click_df.columns.map(str)

    significantly_improved_list = []
    improved_list = []
    for pid in reshaped_click_df:
        result = mk.original_test(reshaped_click_df[pid])
        if exp == "high_variance_low_cost" or exp == "high_variance_high_cost":
            if result.trend == "increasing":
                significantly_improved_list.append(pid)
            if result.s > 0:
                improved_list.append(pid)
        if exp == "low_variance_low_cost" or exp == "low_variance_high_cost":
            if result.trend == "decreasing":
                significantly_improved_list.append(pid)
            if result.s < 0:
                improved_list.append(pid)
    # print(f"Out of {reshaped_click_df.shape[1]} participants, {len(significantly_improved_list)} significantly improved, "
    #       f"({len(significantly_improved_list) / reshaped_click_df.shape[1] * 100}%)")
    print(f"Out of {reshaped_click_df.shape[1]} participants, {len(improved_list)} improved, "
          f"({len(improved_list) / reshaped_click_df.shape[1] * 100}%)")

    # observed = [[len(improved_list), reshaped_click_df.shape[1] - len(improved_list)],
    #             [reshaped_click_df.shape[1] * 0.5, reshaped_click_df.shape[1] * 0.5]]
    # chi2, p_value, dof, _ = chi2_contingency(observed)
    # print(p_value, chi2, dof)


def create_pairs(lst):
    pairs = []
    for i in range(len(lst) - 1):
        if lst[i] != lst[i + 1]:
            pair = (lst[i], lst[i + 1])
            pairs.append(pair)
    return pairs


def monotonous_change(exp, data):
    click_df = data[['pid', 'trial', 'number_of_clicks']].copy()
    reshaped_click_df = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")
    reshaped_click_df.columns = reshaped_click_df.columns.map(str)

    # remove no change pair
    pairs = []
    for columns in reshaped_click_df:
        clicks = reshaped_click_df[columns]
        for i in range(len(clicks) - 1):
            if clicks[i] != clicks[i + 1]:
                pair = (clicks[i], clicks[i + 1])
                pairs.append(pair)

    improvement = 0
    no_improvement = 0

    for tup in pairs:
        if tup[0] < tup[1]:
            improvement += 1
        elif tup[0] >= tup[1]:
            no_improvement += 1

    print(
        f"Out of {improvement + no_improvement} click changes, {improvement} are improvements ({improvement / (improvement + no_improvement) * 100})")
    return None


def pid_improved_clicks_twice(exp, data, bad_pid):
    click_df = data[['pid', 'trial', 'number_of_clicks']].copy()
    reshaped_click_df = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")

    ## remove bad participants?
    reshaped_click_df = reshaped_click_df.drop(bad_pid, axis=1)

    reshaped_click_df.columns = reshaped_click_df.columns.map(str)

    def check_increasing_sequences(lst):
        prev_num = None
        increases = 0
        for num in lst:
            if prev_num is not None and num > prev_num:
                increases += 1
                if increases >= 2:
                    return 1
            prev_num = num
        return 0

    def check_decreasing_sequences(lst):
        prev_num = None
        increases = 0
        for num in lst:
            if prev_num is not None and num > prev_num:
                increases += 1
                if increases >= 2:
                    return 1
            prev_num = num
        return 0

    if exp == "high_variance_high_cost" or exp == "high_variance_low_cost":
        count_increasing = 0
        for col in reshaped_click_df:
            count_increasing += check_increasing_sequences(list(reshaped_click_df[col]))
        print(
            f"{exp}: {count_increasing} participants improved at least twice out of {reshaped_click_df.shape[1]} participants")
    if exp == "low_variance_low_cost" or exp == "low_variance_high_cost":
        count_decreasing = 0
        for col in reshaped_click_df:
            count_decreasing += check_decreasing_sequences(list(reshaped_click_df[col]))
        print(
            f"{exp}: {count_decreasing} participants improved at least twice out of {reshaped_click_df.shape[1]} participants")
    return None


if __name__ == "__main__":
    experiments = ["high_variance_high_cost",
                   "high_variance_low_cost",
                   "low_variance_high_cost",
                   "low_variance_low_cost"]

    # experiments = ["low_variance_low_cost"]
    click_df_all_conditions = []
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
        click_df = create_click_df(data, experiment)

        # filter for participants who clicked at least once
        # good_pid = clicking_pid(click_df, experiment)
        click_df = click_df[click_df["pid"].isin(learning_pid[experiment])]

        ## participants who improved their clicks at least twice
        # pid_improved_clicks_twice(experiment, click_df, bad_pid)

        ## monotonous change
        # monotonous_change(experiment, click_df)

        ## trend test for each pid
        # trend_within_participant(experiment, click_df)

        ## sequential dependence
        # sequential_dependence(click_df)

        ## magnitude of change
        # magnitude_of_change(click_df, experiment)

        ## plot individual clicks 
        # plot_individual_clicks(click_df, experiment)

        # group click_df by trial and get the average clicks
        average_clicks = click_df.groupby(["trial"])["number_of_clicks"].mean()

        ##plot the average clicks
        plot_clicks(average_clicks)

        ##trend test
        # trend_test(average_clicks)

        ##normality test
        # normality_test(average_clicks) #high_variance_low_cost is not normally distributed

        ##append all 4 conditions into one df
        # click_df_all_conditions.append(click_df)

        # optimal number of clicks vs. actual number of clicks
        # get clicks of last trial
        # if experiment == "high_variance_low_cost":
        #     chi2, p = chisquare([average_clicks.values[-1], 7.1])
        #     print(f"chi^ goodness of fit test for {experiment}: s={chi2}, p={p}")
        # elif experiment == "high_variance_high_cost":
        #     chi2, p  = chisquare([average_clicks.values[-1], 6.32])
        #     print(f"chi^ goodness of fit test for {experiment}: s={chi2}, p={p} ")
        # elif experiment == "low_variance_high_cost":
        #     chi2, p  = chisquare([average_clicks.values[-1], 0])
        #     print(f"chi^ goodness of fit test for {experiment}: s={chi2}, p={p} ")
        # else:
        #     chi2, p  = chisquare([average_clicks.values[-1], 5.82])
        #     print(f"chi^ goodness of fit test for {experiment}: s={chi2}, p={p} ")

        # anova(click_df_all_conditions)
    # result_df = pd.concat(click_df_all_conditions, ignore_index=True)
    # lme(result_df)
