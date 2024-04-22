from scipy.stats import ranksums, normaltest, wilcoxon, kendalltau, spearmanr, boxcox, sem, t, chisquare
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
import numpy as np
import json


def plot_score(pretraining_exp, training, test_exp, pretraining_control, test_control, experiment):
    """

    Args:
        pretraining: average pretraining score
        training: average training score
        test:  average posttraining score
        experiment: "with_click" or "no_click"

    Returns:

    """
    # making all into one array
    # x = np.hstack([pretraining, training, test])

    pretraining_exp_mean = np.mean(pretraining_exp)
    test_exp_mean = np.mean(test_exp)
    pretraining_control_mean = np.mean(pretraining_control)
    test_control_mean = np.mean(test_control)

    ci = 1.96 * np.std(training) / np.sqrt(len(training))

    plt.plot(range(1, len(training) + 1), training, label="Training (experimental only)")
    plt.fill_between(range(1, len(training) + 1), training - ci, training + ci, color="b", alpha=.1)

    # add the pre and posttraining exp
    # plt.plot(0, pretraining_exp_mean, 'go', label="Average experimental score")
    plt.errorbar(0, pretraining_exp_mean, yerr=1.96 * np.std(pretraining_exp) / np.sqrt(len(pretraining_exp)), fmt="go", alpha=0.5, label="Average experimental score")

    # plt.plot(31, test_exp_mean, 'go')
    plt.errorbar(31, test_exp_mean, yerr=1.96 * np.std(test_exp) / np.sqrt(len(test_exp)), fmt="go", alpha=0.5)

    # add the pre and posttraining control
    # plt.plot(0, pretraining_control_mean, 'ro', label="Average control score")
    plt.errorbar(0, pretraining_control_mean, yerr=1.96 * np.std(pretraining_control) / np.sqrt(len(pretraining_control)), fmt="ro", alpha=0.5, label="Average control score")

    # plt.plot(31, test_control_mean, 'ro')
    plt.errorbar(31, test_control_mean, yerr=1.96 * np.std(test_control) / np.sqrt(len(test_control)), fmt="ro", alpha=0.5)

    plt.ylim(top=60)
    # plt.axvline(x=0.5)
    # plt.axvline(x=29.5)
    plt.xlabel("Trials", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.title(experiment)
    plt.legend(fontsize=10, loc="lower center")
    # plt.savefig(f"average_score_training_{experiment}_{condition}_journal")
    plt.show()
    plt.close()
    return None


def filter_learning_pid(data_training):
    # if no significant increase in score, then not learning
    # get all pid
    learners = []
    pid_list = data_training["pid"].unique()
    for pid in pid_list:
        score = data_training[data_training["pid"] == pid]["score"]
        mk_test = mk.original_test(score, alpha=0.2)
        # if mk_test[5] > 261:
        #     learners.append(pid)
        if mk_test[0] == "increasing":
            learners.append(pid)
    return learners


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


experiment = "with_click"
condition = "control"

data = pd.read_csv(f"../../data/human/existence_{experiment}_{condition}/mouselab-mdp.csv")
pid = pd.read_csv(f"../../data/human/existence_{experiment}_{condition}/participants.csv")
number_of_participants = len(pid)
print("number_of_participants", number_of_participants)
number_of_training_trial = 30

# filter data by training, test and pretraining
data_training = data[data["block"] == "training"]
data_test = data[data["block"] == "test"]
data_pretraining = data[data["block"] == "pretraining"]

### SCORE DEVELOPMENT DURING TRAINING
if condition == "exp":
    trial_index = list(range(1, number_of_training_trial + 1)) * number_of_participants
    data_training["trial_index"] = trial_index
    average_score = data_training.groupby(["trial_index"])["score"].mean()
    average_score = average_score.reset_index(drop=True)

    # filter series by first 15
    # average_score = average_score[:15]

    #  Mann Kendall test of trend for training
    mk_result = mk.original_test(average_score[0:12])
    print("Score trend test results: ", mk_result)

    # filter by learners
    learners = list(filter_learning_pid(data_training))
    data_training_learners = data_training[data_training["pid"].isin(learners)]

    # Trend test for learners
    average_score_learners = data_training_learners.groupby(["trial_index"])["score"].mean()
    average_score_learners = average_score_learners.reset_index(drop=True)
    mk_result_learners = mk.original_test(average_score_learners)
    print("Score trend test results for learners: ", mk_result_learners)


### DESCRIPTIVE STATISTICS FOR SCORE
print("Pretraining mean score: ", data_pretraining["score"].mean())
print("Pretraining std score: ", data_pretraining["score"].std())
print("Test mean score: ", data_test["score"].mean())
print("Test std score: ", data_test["score"].std())
# # 95% CI for score
# _, lower, upper = mean_confidence_interval(data_pretraining["score"], confidence=0.95)
# print(f"95% CI for the pretraining score: lower {lower}, upper {upper}")
# _, lower, upper = mean_confidence_interval(data_test["score"], confidence=0.95)
# print(f"95% CI for the posttraining score: lower {lower}, upper {upper}")
delta = np.array(data_test["score"]) - np.array(data_pretraining["score"])
_, lower, upper = mean_confidence_interval(delta, confidence=0.95)
print(f"95% CI for the difference: lower {lower}, upper {upper}")

### Wilcoxon tests
if experiment == "no_click":
    ranksum_results = wilcoxon(data_pretraining["score"], data_test["score"], alternative="greater")
    print("Wilcoxon signed-rank test results (pretraining > test score): ", ranksum_results)
else:
    ranksum_results = wilcoxon(data_pretraining["score"], data_test["score"], alternative="less")
    print("Wilcoxon signed-rank test results (pretraining < test score): ", ranksum_results)

### WILCOXON test filtered for learners
if condition == "exp":
    data_pretraining_learners = data_pretraining[data_pretraining["pid"].isin(learners)]
    data_test_learners = data_test[data_test["pid"].isin(learners)]
    print(f"Learners n= {len(learners)} out of {len(data_pretraining)} participants")
    print("Pretraining mean score learners: ", data_pretraining_learners["score"].mean())
    print("Pretraining std score learners: ", data_pretraining_learners["score"].std())
    print("Test mean score learners: ", data_test_learners["score"].mean())
    print("Test std score learners: ", data_test_learners["score"].std())
    # _, lower, upper = mean_confidence_interval(data_pretraining_learners["score"], confidence=0.95)
    # print(f"95% CI for the pretraining score for learners: lower {lower}, upper {upper}")
    # _, lower, upper = mean_confidence_interval(data_test_learners["score"], confidence=0.95)
    # print(f"95% CI for the posttraining score for learners: lower {lower}, upper {upper}")
    # 95% CI for the change
    delta = np.array(data_test_learners["score"]) - np.array(data_pretraining_learners["score"])
    _, lower, upper = mean_confidence_interval(delta, confidence=0.95)
    print(f"95% CI for the difference for learners: lower {lower}, upper {upper}")

    if experiment == "no_click":
        ranksum_results = wilcoxon(np.array(data_pretraining_learners["score"]), np.array(data_test_learners["score"]),
                                   alternative="greater")
        print("Wilcoxon signed-rank test results (pretraining > test score) for learners: ", ranksum_results)
    else:
        ranksum_results = wilcoxon(np.array(data_pretraining_learners["score"]), np.array(data_test_learners["score"]),
                                   alternative="less")
        print("Wilcoxon signed-rank test results (pretraining < test score) for learners: ", ranksum_results)

if experiment == "no_click":
    print(
        f"{len(data_pretraining[data_pretraining['score'].isin([40, 43])])} participants used best path in decreasing variance pretraining")
    print(
        f"{len(data_test[data_test['score'].isin([40, 43])])} participants used best path in decreasing variance posttraining")
    if condition == "exp":
        print(
            f"{len(data_pretraining_learners[data_pretraining_learners['score'].isin([40, 43])])} learners used best path in decreasing variance pretraining")
        print(
            f"{len(data_test_learners[data_test_learners['score'].isin([40, 43])])} learners used best path in decreasing variance posttraining")

        # is the drop significant?
        print(f"Chi^2 test goodness of fit of count in best path chosen for learners: ", chisquare(
            [len(data_pretraining_learners[data_pretraining_learners['score'].isin([40, 43])]),
             len(data_test_learners[data_test_learners['score'].isin([40, 43])])]))

    print(f"Chi^2 test goodness of fit of count in best path chosen : ", chisquare(
        [len(data_pretraining[data_pretraining['score'].isin([40, 43])]),
         len(data_test[data_test['score'].isin([40, 43])])]))
else: # 37 is the maximum score, and -30 the click cost for clicking all final nodes. The next best score is 2 resulting in -28
    print(f"{(data_pretraining['score'] > 6).sum()} participants used best path in increasing variance pretraining")
    print(f"{(data_test['score'] > 6).sum()} participants used best path in increasing variance posttraining")
    if condition == "exp":
        print(
            f"{(data_pretraining_learners['score'] > 6).sum()} learners used best path in increasing variance pretraining")
        print(
            f"{(data_test_learners['score'] > 6).sum()} learners used best path in increasing variance posttraining")

        # is the drop significant?
        print(f"Chi^2 test goodness of fit of count in best path chosen for learners: ",
              chisquare([(data_pretraining_learners['score'] > 6).sum(), (data_test_learners['score'] > 6).sum()]))

    print(f"Chi^2 test goodness of fit of count in best path chosen : ",
          chisquare([(data_pretraining['score'] > 6).sum(), (data_test['score'] > 6).sum()]))

### Wilcoxon test for both experimnetal and control
if condition == "exp":
    data_other_condition = pd.read_csv(f"../../data/human/existence_{experiment}_control/mouselab-mdp.csv")
    pid_other_condition = pd.read_csv(f"../../data/human/existence_{experiment}_control/participants.csv")
else:
    data_other_condition = pd.read_csv(f"../../data/human/existence_{experiment}_exp/mouselab-mdp.csv")
    pid_other_condition = pd.read_csv(f"../../data/human/existence_{experiment}_exp/participants.csv")

data_other_training = data_other_condition[data_other_condition["block"] == "training"]
data_other_test = data_other_condition[data_other_condition["block"] == "test"]
data_other_pretraining = data_other_condition[data_other_condition["block"] == "pretraining"]

# delta_a = np.array(data_test_learners["score"]) - np.array(data_pretraining_learners["score"])  # with click, exp data
delta_a = np.array(data_test["score"]) - np.array(data_pretraining["score"])  # with click, exp data
delta_b = np.array(data_other_test["score"]) - np.array(data_other_pretraining["score"])  # control data

print("mean delta_a: ", delta_a.mean())
print("mean delta_b: ", delta_b.mean())
print("std delta_a: ", delta_a.std())
print("std delta_b: ", delta_b.std())

if experiment == "no_click":
    ranksum_results = ranksums(delta_a, delta_b, alternative="less")
    print("Wilcoxon rank sum test results between the two conditions (delta_exp < delta_control): ", ranksum_results)
else:
    ranksum_results = ranksums(delta_a, delta_b, alternative="greater")
    print("Wilcoxon rank sum test results between the two conditions (delta_exp > delta_control): ", ranksum_results)

if experiment == "no_click":
    drop_exp = len(data_pretraining[data_pretraining['score'].isin([40, 43])]) - len(
        data_test[data_test['score'].isin([40, 43])])
    drop_control = len(data_other_pretraining[data_other_pretraining['score'].isin([40, 43])]) - len(
        data_other_test[data_other_test['score'].isin([40, 43])])
    print(f"Chi^2 test goodness of fit of count in best path chosen difference between exp control ",
          chisquare([drop_exp, drop_control]))

    drop_exp_learners = len(data_pretraining_learners[data_pretraining_learners['score'].isin([40, 43])]) - len(
        data_test_learners[data_test_learners['score'].isin([40, 43])])
    print(f"Chi^2 test goodness of fit of count in best path chosen difference between exp control for learners",
          chisquare([drop_exp_learners, drop_control]))
else:
    increase_exp = (data_test['score'] > 6).sum() - (data_pretraining['score'] > 6).sum()
    increase_control = (data_other_test['score'] > 6).sum() - (data_other_pretraining['score'] > 6).sum()
    print(f"Chi^2 test goodness of fit of count in best path chosen difference between exp control ",
          chisquare([increase_exp, increase_control]))

    increase_exp_learners = (data_test_learners['score'] > 6).sum() - (data_pretraining_learners['score'] > 6).sum()
    print(f"Chi^2 test goodness of fit of count in best path chosen difference between exp control for learners",
          chisquare([increase_exp_learners, increase_control]))

# plot of training
if condition == "exp":
    plot_score(data_pretraining["score"], average_score, data_test["score"],
               data_other_pretraining["score"], data_other_test["score"], experiment)


### create csv for R (robust anova test)
def create_csv_for_r(data_pretraining, data_test, data_other_pretraining, data_other_test, learners=False):
    all_data = pd.DataFrame()
    score = pd.concat([data_pretraining["score"], data_test["score"], data_other_pretraining["score"],
                       data_other_test["score"]], axis=0)

    # create lists for condition
    len_learners = len(data_test_learners)
    len_exp = len(data_test)
    len_control = len(data_other_test)

    # create dummy variables as marker for condition (exp or control) and trial (pre or test)
    if learners:
        condition = [["exp"] * len_learners, ["exp"] * len_learners, ["control"] * len_control,
                     ["control"] * len_control]
        trial = [["pretraining"] * len_learners, ["posttraining"] * len_learners, ["pretraining"] * len_control,
                 ["posttraining"] * len_control]
    else:
        condition = [["exp"] * len_exp, ["exp"] * len_exp, ["control"] * len_control, ["control"] * len_control]
        trial = [["pretraining"] * len_exp, ["posttraining"] * len_exp, ["pretraining"] * len_control,
                 ["posttraining"] * len_control]
    condition_flat = [item for sublist in condition for item in sublist]
    trial_flat = [item for sublist in trial for item in sublist]

    if learners:
        match_pid_a = [[*range(0, len_learners)] * 2]
        match_pid_b = [[*range(len_learners, len_learners + len_control)] * 2]
    else:
        match_pid_a = [[*range(0, len_exp)] * 2]
        match_pid_b = [[*range(len_exp, len_exp + len_control)] * 2]

    match_pid_a_flat = [item for sublist in match_pid_a for item in sublist]
    match_pid_b_flat = [item for sublist in match_pid_b for item in sublist]
    match = [match_pid_a_flat, match_pid_b_flat]
    match_flat = [item for sublist in match for item in sublist]

    all_data["score"] = score
    all_data["condition"] = condition_flat
    all_data["trial"] = trial_flat
    all_data["pid"] = match_flat
    if learners:
        all_data.to_csv(f"{experiment}_scores_learners.csv", index=False)
    else:
        all_data.to_csv(f"{experiment}_scores.csv", index=False)

# create_csv_for_r(data_pretraining, data_test, data_other_pretraining, data_other_test, learners=False)
# create_csv_for_r(data_pretraining_learners, data_test_learners, data_other_pretraining, data_other_test, learners=True)
