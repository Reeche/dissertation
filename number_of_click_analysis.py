import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""
This file contains the new analysis for planning amount experiment for the journal paper submission
Changes compared to NeurIPS submission: 
* Now contains al data, i.e. the participants who clicked nothing throughout all trials are now included

"""

def create_click_df(data):
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
        click_df["click_cost"] = [1] * len(click_temp_df)
        click_df["variance"] = [1] * len(click_temp_df)
    elif experiment == "high_variance_high_cost":
        click_df["click_cost"] = [5] * len(click_temp_df)
        click_df["variance"] = [1] * len(click_temp_df)
    elif experiment == "low_variance_low_cost":
        click_df["click_cost"] = [1] * len(click_temp_df)
        click_df["variance"] = [0] * len(click_temp_df)
    elif experiment == "low_variance_high_cost":
        click_df["click_cost"] = [5] * len(click_temp_df)
        click_df["variance"] = [0] * len(click_temp_df)

    return click_df

def plot_clicks(average_clicks):
    ci = 1.96 * np.std(average_clicks) / np.sqrt(len(average_clicks))
    plt.plot(average_clicks)
    plt.fill_between(range(len(average_clicks)), average_clicks - ci, average_clicks + ci, color="b",
                     alpha=.1)
    plt.ylim(top=9)
    plt.xlabel("Trial Number", size=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if experiment == "high_variance_low_cost":
        label = "HVLC"
        plt.axhline(y=7.10, color='r', linestyle='-')
    elif experiment == "high_variance_high_cost":
        label = "HVHC"
        plt.axhline(y=6.32, color='r', linestyle='-')
    elif experiment == "low_variance_high_cost":
        label = "LVHC"
        plt.axhline(y=0.68, color='r', linestyle='-') #it is actually 0 but needs to show on plot, therefore 0.68
    else:
        label = "LVLC"
        plt.axhline(y=5.82, color='r', linestyle='-')
    plt.ylabel(f"Average number of clicks for {label}", fontsize=14)
    plt.savefig(f"results/plots/{experiment}_average_clicks")
    # plt.show()
    plt.close()
    return None

def trend_test(average_clicks):
    result = mk.original_test(average_clicks)
    print(f"Mann Kendall test for clicks for {experiment}: clicks are {result}")
    return None

def normality_test(average_clicks):
    k2, p = stats.normaltest(average_clicks)
    print(f"Normality test for clicks for {experiment}: p = {p}")

def anova(click_data):
    model = ols('number_of_clicks ~ C(trial) + C(click_cost) + C(pid) + C(variance)', data=click_data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    print(table)
    return None

def glm(click_data):
    y = click_data["number_of_clicks"]
    x_temp = click_data.drop(columns=['number_of_clicks', 'clicks'])
    # data = sm.datasets.scotland.load()
    x = sm.add_constant(x_temp)
    gamma_model = sm.GLM(y, x, family=sm.families.Gamma())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())
    return None

if __name__ == "__main__":
    experiments = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost", "low_variance_high_cost"]

    click_df_all_conditions = pd.DataFrame()
    for experiment in experiments:
        data = pd.read_csv(f"data/human/{experiment}/mouselab-mdp.csv")
        click_df = create_click_df(data)

        # group click_df by trial and get the average clicks
        average_clicks = click_df.groupby(["trial"])["number_of_clicks"].mean()

        # plot the average clicks
        # plot_clicks(average_clicks)

        # trend test
        # trend_test(average_clicks)

        # normality test
        # normality_test(average_clicks) #high_variance_low_cost is not normally distributed

        # append all 4 conditions into one df
        click_df_all_conditions = click_df_all_conditions.append(click_df)

    anova(click_df_all_conditions)
    glm(click_df_all_conditions)
