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
from scipy.stats import chisquare

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



def clicking_pid(click_df, experiment):
    pid_list = click_df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
        if any(v != 0 for v in temp_list):
            good_pid.append(pid)
    print(f"{experiment} number of people who clicked something ", len(good_pid), "out of ", len(pid_list))
    print(good_pid)


if __name__ == "__main__":
    experiments = ["v1.0", "c2.1", "c1.1"]
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
        click_df = create_click_df(data)

        ## no clicking pid
        clicking_pid(click_df, experiment)