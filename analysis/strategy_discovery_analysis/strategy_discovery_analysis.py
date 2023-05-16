import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import chisquare, sem, t

data = pd.read_csv(f"../../data/human/strategy_discovery/mouselab-mdp.csv")


### create df
def create_df(data):
    # drop pid 22, 41
    # check if pid has more than 119 trials
    pid_list = data["pid"].unique()
    i = 0
    for pid in pid_list:
        # remove the pid with more than 119 trials
        temp = data[data["pid"] == pid]
        if len(temp) > 120:
            i += 1
            data = data[data["pid"] != pid]

    click_df = pd.DataFrame(columns=["pid", "trial", "number_of_clicks", "clicks", "score", "optimal_strategy"])

    # create a list of all pid * number of trials and append their clicks
    click_df["pid"] = data["pid"]
    # click_df["trial"] = data["trial_index"]
    click_df["score"] = data["score"]

    # get their number of clicks
    number_of_clicks_list = []
    clicks_list = []
    click_temp_df = data["queries"]
    for index, row in click_temp_df.items():
        temp = ast.literal_eval(row)
        clicks = temp["click"]["state"]["target"]
        clicks_list.append(clicks)
        len_of_clicks = len(clicks)
        number_of_clicks_list.append(len_of_clicks)

    click_df["clicks"] = clicks_list
    click_df["number_of_clicks"] = number_of_clicks_list

    # reindex trial count because some participants had to redo the quiz
    trial_number = list(range(1, 121))
    trial_number = trial_number * len(pid_list) # (max(data["pid"]) - i)
    click_df["trial"] = trial_number

    return pid_list, click_df


pid_list, click_df = create_df(data)

# optimal strategy is to first click inner nodes and then outer nodes
immediate_nodes = [1, 5, 9]
middle_nodes = [2, 6, 10]
outer_nodes = [3, 4, 7, 8, 11, 12]

for idx, row in click_df.iterrows():
    # if row["number_of_clicks"] > 1 and row["number_of_clicks"] < 5: #relax to second most optimal strategy by < 6
    #     row["clicks"] = [int(i) for i in row["clicks"]]
        # if all but last one clicks are immediate nodes and last click is the outer node
        # if all(item in immediate_nodes for item in row["clicks"][:-1]) and \
        #         all(item in outer_nodes for item in row["clicks"][-1:]) and row["score"] > 34:
    if row["score"] in [15, 14, 13]:
    # if row["score"] in [35, 34, 33]:
        click_df.at[idx, 'optimal_strategy'] = True
    else:
        click_df.at[idx, 'optimal_strategy'] = False

# calculate proportion
click_df = click_df[click_df["optimal_strategy"] == True]
count = click_df.groupby("trial")["optimal_strategy"].count()
count_prop = count/len(pid_list)

def credible_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

print("CI: ", credible_interval(count_prop[-10:]))

def chi_square_test(count_prop):
    # need actual count for goodness of fit test
    final_prop = count_prop.values[-1]
    res = chisquare([final_prop * 100, (1-final_prop) * 100], f_exp=[0.15 * 100, 0.85 * 100])
    print("Chi^2 goodness of fit: ", res)

chi_square_test(count_prop)

def trend_test(count_prop):
    # increasing until when?
    # for i in range(2, 121):
    #     # print(count_prop[:i])
    #     result = mk.original_test(count_prop[-i:])
    #     if result[0] == "no trend":
    #         print(i)
    result_all = mk.original_test(count_prop)
    print(result_all)
    return None

# trend_test(count_prop)

def plot_proportion(count_prop):
    plt.ylabel("Proportion of optimal strategy")
    plt.xlabel("Trials")
    # reindex count_prop
    count_prop = count_prop.reset_index(drop=True)
    ci = 0.1 * np.std(count_prop) / np.mean(count_prop)
    plt.plot(count_prop)
    x = list(range(0, len(count_prop)))
    plt.fill_between(x, (count_prop-ci), (count_prop+ci), color='blue', alpha=0.1)
    plt.savefig("strategy_discovery.png")
    plt.show()
    # plt.close()
    return None

# plot_proportion(count_prop)

