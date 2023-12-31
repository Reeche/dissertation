import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import chisquare, sem, t, norm
import random
import statsmodels.formula.api as smf
import os
import statsmodels.api as sm


### create df
def create_df(data, learning_pid=None):
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
    trial_number = trial_number * len(pid_list)  # (max(data["pid"]) - i)
    click_df["trial"] = trial_number

    # filter for learning pid
    if learning_pid:
        click_df = click_df[click_df["pid"].isin(learning_pid)]

    return pid_list, click_df


def credible_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def chi_square(count_prop):
    # need actual count for goodness of fit test
    final_prop = count_prop.values[-1]
    res = chisquare([final_prop * 100, (1 - final_prop) * 100], f_exp=[0.15 * 100, 0.85 * 100])
    print("Chi^2 goodness of fit: ", res)


def trend(count_prop):
    # increasing until when?
    # for i in range(2, 121):
    #     # print(count_prop[:i])
    #     result = mk.original_test(count_prop[-i:])
    #     if result[0] == "no trend":
    #         print(i)
    result_all = mk.original_test(count_prop)
    print(result_all)
    return None


def plot_proportion(count_prop):
    # this was used for the participants but I think something is wrong here, CI goes negative
    plt.ylabel("Proportion of optimal strategy")
    plt.xlabel("Trials")
    # reindex count_prop
    count_prop = count_prop.reset_index(drop=True)
    ci = 0.1 * np.std(count_prop) / np.mean(count_prop)
    plt.plot(count_prop)
    x = list(range(0, len(count_prop)))
    plt.fill_between(x, (count_prop - ci), (count_prop + ci), color='blue', alpha=0.1)
    # plt.savefig("strategy_discovery.png")
    plt.legend()
    plt.show()
    plt.close()
    return None


def plot_confidence_interval(actual_count, prop_counts, model, participant=None):
    # Calculate mean and standard deviation of counts
    ci = 1.96 * np.std(prop_counts) / np.sqrt(len(prop_counts))

    # Plot the counts as a line plot
    x = np.arange(len(prop_counts))
    plt.plot(x, prop_counts, label=model, color='blue')
    plt.fill_between(x, prop_counts - ci, prop_counts + ci, alpha=0.3, label=f'95% Confidence Interval')

    if participant:
        ci_pid = 1.96 * np.std(participant[1]["optimal_strategy"]) / np.sqrt(len(prop_counts))
        plt.plot(x, participant[1]["optimal_strategy"], label='Participant', color='red')
        plt.fill_between(x, participant[1]["optimal_strategy"] - ci_pid, participant[1]["optimal_strategy"] + ci_pid,
                         alpha=0.3, label=f'95% Confidence Interval')

    # Set plot labels and legend
    plt.xlabel('Index')
    plt.ylabel('Proportion')
    plt.legend()
    plt.savefig(f"{model}.png")
    # plt.show()
    plt.close()


def classify_via_score(click_df, pid_list):
    # given the score, calculate the proportion of participants who used the optimal strategy
    for idx, row in click_df.iterrows():
        if row["score"] in [15, 14, 13]:  # does 16 make sense?
            click_df.at[idx, 'optimal_strategy'] = True
        else:
            click_df.at[idx, 'optimal_strategy'] = False

    # calculate proportion
    # click_df = click_df[click_df["optimal_strategy"] == True]
    count = click_df.groupby('trial')['optimal_strategy'].sum().astype(int).reset_index()

    # count = click_df.groupby("trial")["optimal_strategy"].count()
    count_prop = count / len(pid_list)
    return count, count_prop


def classify_sequences(sequences):
    # todo: not working correctly
    # optimal strategy is to first click inner nodes and then outer nodes
    immediate_nodes = [1, 5, 9]
    middle_nodes = [2, 6, 10]
    outer_nodes = [3, 4, 7, 8, 11, 12]

    misclassified_sequences = []
    for sequence in sequences:
        if len(sequence) > 1 and len(sequence) < 6:  # relax to second most optimal strategy by < 6
            # if all but last one clicks are immediate nodes and last click is the outer node
            if sequence[0] in immediate_nodes and sequence[-1] in outer_nodes:
                misclassified_sequences.append(sequence)

    print(len(misclassified_sequences) / len(sequences))
    return len(misclassified_sequences) / len(sequences)


def generate_random_sequences():
    sequences = []
    for _ in range(100000):  # 100.000 gives the same result as 1.000.000
        length = random.randint(0, 12)
        random_list = [random.randint(1, 12) for _ in range(length)]
        sequences.append(random_list)
    return sequences


def random_sequences():
    ## check how many randomly generated click sequences are wrongly classified as adaptive ones
    random_sequences = generate_random_sequences()
    misclassified_prop = classify_via_score(random_sequences)
    res = chisquare([0.4 * 100, 0.6 * 100], f_exp=[misclassified_prop * 100, (1 - misclassified_prop) * 100])
    print(res)


def score_trend(click_df):
    click_df = click_df[['pid', 'trial', 'score']].copy()
    reshaped_score_df = click_df.pivot(index="trial", columns="pid", values="score")
    reshaped_score_df.columns = reshaped_score_df.columns.map(str)

    count = 0
    for pid in reshaped_score_df:
        result = mk.original_test(reshaped_score_df[pid])
        if result.s > 0:
            count += 1

    print(f"{count} out of {reshaped_score_df.shape[0]} ({count / reshaped_score_df.shape[0]}) participants improved ")
    return None


def linear_regression(data):
    # linear regression analysis of the score
    formula_ = "score ~ trial"
    res = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
    print(res.summary())


def logistic_regression(data):
    # logitic regression of the proportion of optimal strategy
    # filter data, only keep columns pid, trial, optimal_strategy
    data = data[["pid", "trial", "optimal_strategy"]].copy()

    # replace true and false with 1 and 0
    data["optimal_strategy"] = data["optimal_strategy"].astype(int)

    res = sm.GLM.from_formula("optimal_strategy ~ trial", data=data, family=sm.families.Binomial()).fit()
    print(res.summary())
    return None


def improving_pid(data):
    # pivot table by pid and trial_index
    data = data[['pid', 'trial_index', 'score']].copy()

    # reset trial_index for each pid from 0 to 119
    data["trial_index"] = data.groupby("pid").cumcount()

    reshaped_score_df = data.pivot(index="trial_index", columns="pid", values="score")

    # Mann Kendall test and check if test statistic is positive, then add the pid to adaptiv elist
    adaptive_pid = []
    for pid in reshaped_score_df:
        # result = mk.original_test(reshaped_score_df[pid])
        # if result.s > 0:
        adaptive_pid.append(pid)
    print("Adaptive pid", adaptive_pid)
    return None


def pid_with_adaptive_strategy(data):
    # find the list of participants who used the adaptive strategy in the last 10 trials
    data = data[['pid', 'trial_index', 'score']].copy()

    # reset trial_index for each pid from 0 to 119
    data["trial_index"] = data.groupby("pid").cumcount()

    reshaped_score_df = data.pivot(index="trial_index", columns="pid", values="score")

    average_score_pid = []
    # check if score is > 12 in the last trials
    for pid in reshaped_score_df:
        if reshaped_score_df[pid][-20:].mean() > 12:
            average_score_pid.append(pid)
    print("PID whose average score > 12 in the last n trials", average_score_pid)
    print(len(average_score_pid))

    adaptive_strategy_pid = []
    # check if someone has a score of 15, 14, 13, at least n times
    for pid in reshaped_score_df:
        if reshaped_score_df[pid].isin([13, 14, 15]).sum() >= 10:
            adaptive_strategy_pid.append(pid)
    print("PID who used adaptive strategy at least 3 times", adaptive_strategy_pid)
    print(len(adaptive_strategy_pid))
    return adaptive_strategy_pid


def infer_fitted_model_strategy(model):
    # get all the files in the directory that has the model number
    files = os.listdir("../../results_rl_variants_8000/mcrl/strategy_discovery_data/")
    matching_files = [file for file in files if str(model) in file]

    # create df
    model_df = pd.DataFrame(columns=["pid", "trial", "score"])
    temp_score = []
    temp_pid = []
    temp_trial = []
    for fil in matching_files:
        data = pd.read_pickle(f"../../results_rl_variants_8000/mcrl/strategy_discovery_data/{fil}")
        temp_score.append(data["r"][0])
        temp_pid.append([fil.split("_")[0]] * 120)
        temp_trial.append(range(1, 121))
    model_df["pid"] = [item for sublist in temp_pid for item in sublist]
    model_df["trial"] = [item for sublist in temp_trial for item in sublist]
    model_df["score"] = [item for sublist in temp_score for item in sublist]

    return classify_via_score(model_df, matching_files)


def clicking_pid(click_df):
    pid_list = click_df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
        if any(v > 0 for v in temp_list):
            good_pid.append(pid)
    return good_pid


if __name__ == "__main__":
    data = pd.read_csv(f"../../data/human/strategy_discovery/mouselab-mdp.csv")

    learning_pid = improving_pid(data)  # todo: currently set to all participants
    # sig_improving_pid = pid_with_adaptive_strategy(data)

    pid_list, click_df = create_df(data, learning_pid)

    clicked_pid = clicking_pid(click_df)

    ### analysis
    linear_regression(click_df)
    logistic_regression(click_df)
    # score_trend(click_df)
    # count_prop = classify_via_score(click_df, pid_list)

    # print("CI: ", credible_interval(count_prop[-10:]))
    # trend_test(count_prop)
    # chi_square_test(count_prop)
    # plot_confidence_interval(count_prop_pid[0]["optimal_strategy"], count_prop_pid[1]["optimal_strategy"], "Participant")

    ### classify model strategy
    # vanilla models only
    # models = [522, 491, 479, 1743, 1756]

    # reinforce variants without hierarchical models
    # models = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]
    # models = [491]
    # for model in models:
    #     actual_count, count_prop = infer_fitted_model_strategy(model)
    #     plot_confidence_interval(actual_count["optimal_strategy"], count_prop["optimal_strategy"], model, count_prop_pid)
