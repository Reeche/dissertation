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
from vars import clicked_pid, assign_model_names


### create df
def create_df(data):
    pid_list = data["pid"].unique()
    # reset trial for each participant to start from 0
    data["trial_index"] = data.groupby("pid").cumcount()

    # remove the pid with more than 119 trials
    temp = data.groupby("pid")["trial_index"].max()
    temp = temp[temp < 120]
    data = data[data["pid"].isin(temp.index)]

    click_df = pd.DataFrame(columns=["pid", "trial", "number_of_clicks", "clicks", "score", "optimal_strategy"])

    # create a list of all pid * number of trials and append their clicks
    click_df["pid"] = data["pid"]
    click_df["trial"] = data["trial_index"]
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

    # if the score is 15, 14, 13, then optimal strategy is true
    click_df["optimal_strategy"] = click_df["score"].isin([15, 14, 13])

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
    result_all = mk.original_test(count_prop[1]["optimal_strategy"])
    print(result_all)
    return None


def plot_proportion(count_prop):
    # this was used for the participants but I think something is wrong here, CI goes negative


    # reindex count_prop
    count_prop = count_prop.reset_index(drop=True)
    ci = 0.1 * np.std(count_prop) / np.mean(count_prop)
    plt.plot(count_prop[1]["optimal_strategy"], label="Participant", color='red')
    x = list(range(0, len(count_prop)))
    plt.fill_between(x, (count_prop - ci), (count_prop + ci), color='blue', alpha=0.1)

    # font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # legend font size
    plt.legend(fontsize=12, loc='lower right')
    plt.ylabel("Proportion of optimal strategy")
    plt.xlabel("Trials")

    plt.savefig("plots/strategy_discovery.png")

    # plt.show()
    plt.close()
    return None


def plot_confidence_interval(actual_count, prop_counts, model, participant=None):
    # turn prop_copunt into percentages
    # prop_counts = prop_counts * 100

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
    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Proportion of adaptive strategy', fontsize=12)
    plt.legend()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')

    # plt.savefig(f"plots/proportion.png")
    plt.show()
    plt.close()


def classify_via_score(click_df):
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
    count_prop = count / len(clicked_pid)
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

    ### mean trend
    # calculate the mean score for each trial
    mean_score = reshaped_score_df.mean(axis=1)
    # mk test of trend
    result = mk.original_test(mean_score)
    print("Mean score", result)

    count = 0
    for pid in reshaped_score_df:
        result = mk.original_test(reshaped_score_df[pid])
        if result.s > 0:
            count += 1

    print(f"{count} out of {reshaped_score_df.shape[0]} ({click_df['pid'].unique()}) participants improved ")
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
    # print("Adaptive pid", adaptive_pid)
    return adaptive_pid


def pid_with_adaptive_strategy(data):
    # find the list of participants who used the adaptive strategy in the last 10 trials
    data = data[['pid', 'trial_index', 'score']].copy()

    # reset trial_index for each pid from 0 to 119
    data["trial_index"] = data.groupby("pid").cumcount()

    reshaped_score_df = data.pivot(index="trial_index", columns="pid", values="score")

    average_score_pid = []
    # check if score is > 12 in the last trials
    for pid in reshaped_score_df:
        if 16 > reshaped_score_df[pid][-1:].mean() > 12:
            average_score_pid.append(pid)
    print("PID whose average score > 12 in the last n trials", average_score_pid)
    print(len(average_score_pid))

    adaptive_strategy_pid = []
    # check if someone has a score of 15, 14, 13, at least n times regardless of when
    for pid in reshaped_score_df:
        if reshaped_score_df[pid].isin([13, 14, 15]).sum() >= 3:
            adaptive_strategy_pid.append(pid)
    print("PID who used adaptive strategy at least 3 times", adaptive_strategy_pid)
    print(len(adaptive_strategy_pid))
    return adaptive_strategy_pid


def infer_fitted_model_strategy_server(model):
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


def infer_fitted_model_strategy_local_via_click_sequence(model):
    # it is not as accurate as classifying via score because someone could have clicked e.g 9 and 4, which is the outer node on a different branch
    # therefore not really used
    data = pd.read_csv(f"../likelihood_vanilla_model_comparison/strategy_discovery_1756_lr0.csv")
    data = data[data["model"] == model]

    # keep only pid, model, model_clicks columns
    data = data[["pid", "model", "model_clicks", "pid_clicks"]]

    # get model clicks
    data['model_clicks'] = data['model_clicks'].apply(ast.literal_eval)
    data['pid_clicks'] = data['pid_clicks'].apply(ast.literal_eval)

    # explode model_clicks and pid_clicks
    data = data.explode(['model_clicks', 'pid_clicks']).reset_index(drop=True)

    # add trial index
    data["trial_index"] = data.groupby("pid").cumcount()

    # remove the last item in the list in model_clicks
    data["model_clicks"] = data["model_clicks"].apply(lambda x: x[:-1])
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: x[:-1])

    # get the number of clicks
    data["model_number_of_clicks"] = data["model_clicks"].apply(lambda x: len(x) - 1)
    data["pid_number_of_clicks"] = data["pid_clicks"].apply(lambda x: len(x) - 1)

    # if number of clicks between 4 and 2, then optimal number is true
    data["model_optimal_number"] = data["model_number_of_clicks"].isin([4, 3, 2])
    data["pid_optimal_number"] = data["pid_number_of_clicks"].isin([4, 3, 2])

    # if number of clicks > 1 and if last click is outer node (3, 4, 7, 8, 11, 12)
    data["model_outer_node"] = data["model_clicks"].apply(
        lambda x: x[-1] in [3, 4, 7, 8, 11, 12] if len(x) > 1 else False)
    data["pid_outer_node"] = data["pid_clicks"].apply(lambda x: x[-1] in [3, 4, 7, 8, 11, 12] if len(x) > 1 else False)

    # add column with boolean if all other clicks are inner nodes (1, 5, 9)
    data["model_inner_nodes"] = data["model_clicks"].apply(
        lambda x: all(i in [1, 5, 9] for i in x[:-1] if len(x) > 1) if len(x) > 1 else False)
    data["pid_inner_nodes"] = data["pid_clicks"].apply(
        lambda x: all(i in [1, 5, 9] for i in x[:-1] if len(x) > 1) if len(x) > 1 else False)

    # get another column if the last click is outer node and all other clicks are inner nodes
    data["model_optimal_strategy"] = data["model_outer_node"] & data["model_inner_nodes"] & data["model_optimal_number"]
    data["pid_optimal_strategy"] = data["pid_outer_node"] & data["pid_inner_nodes"] & data["pid_optimal_number"]

    # count the number of participants who used the optimal strategy by trial
    model_count = data.groupby('trial_index')['model_optimal_strategy'].sum().astype(int).reset_index()
    pid_count = data.groupby('trial_index')['pid_optimal_strategy'].sum().astype(int).reset_index()

    # drop the columns that are not needed
    data = data.drop(
        columns=["model_inner_nodes", "pid_inner_nodes", "model_outer_node", "pid_outer_node", "model_optimal_number",
                 "pid_optimal_number", "model_number_of_clicks", "pid_number_of_clicks"])

    # mk test of trend
    result = mk.original_test(model_count["model_optimal_strategy"])
    print(model, result)

    plt.plot(model_count["model_optimal_strategy"] / len(data["pid"].unique()))
    # plt.plot(pid_count["pid_optimal_strategy"] / len(data["pid"].unique()))
    plt.show()
    plt.close()

    return data


def infer_fitted_model_strategy_local_score():
    data = pd.read_csv(f"../../final_results/aggregated_data/strategy_discovery.csv")
    data['model'] = data.apply(assign_model_names, axis=1)
    models = list(data["model"].unique())

    data = data[data["pid"].isin(clicked_pid)]
    data = data[["pid", "model", "model_rewards", "pid_rewards"]]

    for model in models:
        filtered_data = data[data["model"] == model]
        # keep only pid, model, model_clicks columns

        # get model clicks
        filtered_data['model_rewards'] = filtered_data['model_rewards'].apply(ast.literal_eval)
        filtered_data['pid_rewards'] = filtered_data['pid_rewards'].apply(
            lambda s: [int(num) for num in s.strip('[]').split()])

        filtered_data = filtered_data.explode(['model_rewards', 'pid_rewards']).reset_index(drop=True)

        filtered_data["trial_index"] = filtered_data.groupby("pid").cumcount()

        # if the score is 15, 14, 13, then optimal strategy is true
        filtered_data["model_optimal_strategy"] = filtered_data["model_rewards"].isin([15, 14, 13])
        filtered_data["pid_optimal_strategy"] = filtered_data["pid_rewards"].isin([15, 14, 13])

        # count the number of participants who used the optimal strategy by trial
        model_count = filtered_data.groupby('trial_index')['model_optimal_strategy'].sum().astype(int).reset_index()
        pid_count = filtered_data.groupby('trial_index')['pid_optimal_strategy'].sum().astype(int).reset_index()

        # mk test of trend
        result = mk.original_test(model_count["model_optimal_strategy"])
        print(model, result)

        # plt.plot(model_count["model_optimal_strategy"] / len(filtered_data["pid"].unique()), label=model)
        # plt.plot(pid_count["pid_optimal_strategy"] / len(data["pid"].unique()), label="Participant")
        # plt.xlabel("Trials")
        # plt.ylabel("Proportion of optimal strategy")
        # plt.legend()
        # plt.savefig(f"plots/{model}.png")
        # plt.show()
        # plt.close()
    # return data


def clicking_pid(click_df):
    pid_list = click_df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = click_df[click_df['pid'] == pid]["number_of_clicks"].to_list()
        if any(v > 0 for v in temp_list):
            good_pid.append(pid)
    return good_pid

def plot_score(click_df):
    pid_average = click_df.groupby('trial')['score'].mean()

    # Calculate mean and standard error for each data point
    std_dev = np.std(click_df["score"], axis=0)
    n = len(pid_average)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = click_df.groupby('trial')['score'].mean().index

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue")
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trials")
    plt.ylabel("Average score")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    # plt.savefig("plots/score.png")
    plt.show()
    plt.close()
    return None


def plot_combined(click_df, actual_count, prop_counts, model, participant=None):
    # Plot the score development across trials on the primary y-axis
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Average score per trial
    pid_average = click_df.groupby('trial')['score'].mean()
    std_dev = np.std(click_df["score"], axis=0)
    n = len(pid_average)
    std_err = std_dev / np.sqrt(n)
    conf_interval = 1.96 * std_err
    x = click_df.groupby('trial')['score'].mean().index

    ax1.set_xlabel('Trial', fontsize=14)
    ax1.set_ylabel('Average Score', color='blue', fontsize=14)
    ax1.plot(x, pid_average, label="Average Score", color='blue')
    ax1.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot the proportion of adaptive strategies on the secondary y-axis
    ax2 = ax1.twinx()
    ci = 1.96 * np.std(prop_counts) / np.sqrt(len(prop_counts))
    x = np.arange(len(prop_counts))


    ax2.set_ylabel('Proportion of adaptive strategies', color='green', fontsize=14)
    ax2.plot(x, prop_counts, label="Proportion", color='green')
    ax2.fill_between(x, prop_counts - ci, prop_counts + ci, color='green', alpha=0.3, label='95% CI')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(-0.01, 0.5)

    # If participant data is provided, plot it as well; might not need but kept it nonetheless (chatgpt wrote this)
    # if participant:
    #     ci_pid = 1.96 * np.std(participant[1]["optimal_strategy"]) / np.sqrt(len(prop_counts))
    #     ax2.plot(x, participant[1]["optimal_strategy"], label='Participant', color='green')
    #     ax2.fill_between(x, participant[1]["optimal_strategy"] - ci_pid, participant[1]["optimal_strategy"] + ci_pid,
    #                      color='green', alpha=0.3, label='95% CI')

    # Legends and title
    fig.tight_layout()

    # legend for ax1 and ax2 in the same box
    # Get handles and labels from each axis
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Place a combined legend on ax1 (or fig if you want it outside the axes)
    ax1.legend(handles, labels, loc='lower right', fontsize=14)

    plt.savefig(f"plots/cogsci2025/score_proportion.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    data = pd.read_csv(f"../../data/human/strategy_discovery/mouselab-mdp.csv")

    # pid_with_adaptive_strategy_in_last_trials = pid_with_adaptive_strategy(data)

    pid_list, click_df = create_df(data)

    ## filter for clicked_pid
    click_df = click_df[click_df["pid"].isin(clicked_pid)]

    ### analysis
    # linear_regression(click_df)
    # logistic_regression(click_df)
    # score_trend(click_df)

    count_prop = classify_via_score(click_df)

    # plot_score(click_df)
    # plot_confidence_interval(count_prop[0]["optimal_strategy"], count_prop[1]["optimal_strategy"], "Participant")
    plot_combined(click_df, count_prop[0]["optimal_strategy"], count_prop[1]["optimal_strategy"], "Participant", count_prop)

    # print("CI: ", credible_interval(count_prop[-10:]))
    # trend(count_prop)
    # chi_square_test(count_prop)
    # plot_confidence_interval(count_prop[0]["optimal_strategy"], count_prop[1]["optimal_strategy"], "Participant")

    ### classify model strategy
    # test = infer_fitted_model_strategy_local_via_click_sequence()
    # infer_fitted_model_strategy_local_score()
