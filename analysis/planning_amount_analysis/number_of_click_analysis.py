import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from scipy.stats import chisquare

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
    plt.ylim(top=7.5)
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
        plt.axhline(y=0.01, color='r', linestyle='-')  # it is actually 0 but needs to show on plot, therefore 0.68
    else:
        label = "LVLC"
        plt.axhline(y=5.82, color='r', linestyle='-')
    plt.ylabel(f"Average number of clicks for {label}", fontsize=15)
    plt.savefig(f"results/plots/{experiment}_average_clicks")
    # plt.show()
    plt.close()
    return None

def plot_individual_clicks(click_df):
    pid_list = click_df["pid"].unique()

    for pid in pid_list:
        temp_df = click_df[click_df['pid'] == pid]["number_of_clicks"]

        print(2)
        # plt.plot(click_df[click_df['pid'] = pid]['number_of_clicks'])
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

def glm(click_data):
    # filter for high and low variance
    click_data = click_data[click_data["variance"] == 0]

    # click_data["trial_variance"] = click_data["trial"] * click_data["variance"]
    # # click_data["trial:pid"] = click_data["trial"] * click_data["variance"]
    # click_data["trial_cost"] = click_data["trial"] * click_data["click_cost"]
    # click_data["variance_cost"] = click_data["variance"] * click_data["click_cost"]
    # click_data["trial_variance_cost"] = click_data["trial"] * click_data["click_cost"] * click_data["variance"]

    # cutoff = 35
    # # create df with first n trials
    # #x_temp = click_data.drop(columns=['number_of_clicks', 'clicks'])
    # x_learning = click_data[click_data["trial"].isin(range(1,cutoff))]
    # y_learning = x_learning["number_of_clicks"]
    # x_learning = x_learning.drop(columns=['number_of_clicks', 'clicks', 'pid'])
    click_data = sm.add_constant(click_data)

    # create df with last n:35 trials
    # x_nonlearning = click_data[click_data["trial"].isin(range(cutoff,35))]
    # y_nonlearning = x_nonlearning["number_of_clicks"]
    # x_nonlearning = x_nonlearning.drop(columns=['number_of_clicks', 'clicks'])
    # x_nonlearning = sm.add_constant(x_nonlearning)

    # linear mixed effect models
    formula_ = "number_of_clicks ~ C(variance) + trial + trial:C(variance) + click_cost + trial:click_cost + trial:C(variance):click_cost + C(variance):click_cost"
    gamma_model = smf.mixedlm(formula=formula_, data=click_data, groups=click_data["pid"]).fit()  # makes sense

    # glm
    # formula = "number_of_clicks ~ C(pid) + trial + C(variance) + click_cost + trial:C(variance) + trial:click_cost + C(variance):click_cost + trial:C(variance):click_cost"
    # formula = "number_of_clicks ~ trial + C(variance) + click_cost + trial:C(variance) + trial:click_cost + C(variance):click_cost + trial:C(variance):click_cost"
    # gamma_model = smf.glm(formula=formula, data=click_data, family=sm.families.NegativeBinomial()).fit() #does not make sense
    # gamma_model = sm.GLM(y_learning, x_learning, family=sm.families.Poisson()).fit() #poisson makes sense

    # ols
    # gamma_model = sm.OLS(y_learning, x_learning, data=click_data).fit() #makes half sense

    # generalised linear mixed effect models
    # gamma_model = sm.PoissonBayesMixedGLM(endog=y_learning, exog=x_learning, exog_vc=x_learning["pid"], ident=[0])
    # # gamma_model = gpb.GPModel(group_data=click_data, likelihood="binary")
    # gamma_model.fit(y=y_learning, X=x_learning)

    print("learning results", gamma_model.summary())

    # non-learning phase
    # gamma_model = sm.GLM(y_nonlearning, x_nonlearning, family=sm.families.Gamma())
    # gamma_results = gamma_model.fit()
    # print("nonlearning results", gamma_results.summary())
    return None


if __name__ == "__main__":
    experiments = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost",
                   "low_variance_high_cost"]

    # experiments = ["low_variance_high_cost"]
    click_df_all_conditions = pd.DataFrame()
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
        click_df = create_click_df(data)

        ## plot individual clicks 
        plot_individual_clicks(click_df)

        # group click_df by trial and get the average clicks
        average_clicks = click_df.groupby(["trial"])["number_of_clicks"].mean()

        ##plot the average clicks
        # plot_clicks(average_clicks)

        ##trend test
        # trend_test(average_clicks)

        ##normality test
        # normality_test(average_clicks) #high_variance_low_cost is not normally distributed

        ##append all 4 conditions into one df
        click_df_all_conditions = click_df_all_conditions.append(click_df)

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
    glm(click_df_all_conditions)
