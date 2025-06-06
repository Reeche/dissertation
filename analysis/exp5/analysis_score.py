import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ranksums, chisquare, t, sem
import pymannkendall as mk
import os


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    se_val = sem(data)
    h = se_val * t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - h, mean + h


def filter_learning_pid(data_training):
    learners = []
    for pid in data_training["pid"].unique():
        score_series = data_training[data_training["pid"] == pid]["score"]
        mk_test = mk.original_test(score_series, alpha=0.2)
        if mk_test.trend == "increasing":
            learners.append(pid)
    return learners


def plot_score(pretraining_exp, training, test_exp, pretraining_control, test_control, experiment):
    def ci_bounds(data):
        return 1.96 * np.std(data) / np.sqrt(len(data))

    trial_range = range(1, len(training) + 1)
    plt.plot(trial_range, training, label="Training (experimental only)")
    plt.fill_between(trial_range, training - ci_bounds(training), training + ci_bounds(training), color="b", alpha=.1)

    # Experimental pre and post
    plt.errorbar(0, np.mean(pretraining_exp), yerr=ci_bounds(pretraining_exp), fmt="go", alpha=0.5,
                 label="Avg. experimental score")
    plt.errorbar(31, np.mean(test_exp), yerr=ci_bounds(test_exp), fmt="go", alpha=0.5)

    # Control pre and post
    plt.errorbar(0, np.mean(pretraining_control), yerr=ci_bounds(pretraining_control), fmt="ro", alpha=0.5,
                 label="Avg. control score")
    plt.errorbar(31, np.mean(test_control), yerr=ci_bounds(test_control), fmt="ro", alpha=0.5)

    plt.xlabel("Trials", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, loc="lower center")
    plt.ylim(top=60)
    plt.show()
    plt.close()


def create_csv_for_r(pre, post, pre_ctrl, post_ctrl, experiment, learners=False):
    df = pd.DataFrame()
    scores = pd.concat([pre["score"], post["score"], pre_ctrl["score"], post_ctrl["score"]])

    if learners:
        len_exp = len(post)
        len_ctrl = len(post_ctrl)
        conds = [["exp"] * len_exp, ["exp"] * len_exp, ["control"] * len_ctrl, ["control"] * len_ctrl]
        trials = [["pretraining"] * len_exp, ["posttraining"] * len_exp, ["pretraining"] * len_ctrl,
                  ["posttraining"] * len_ctrl]
        pids = [[*range(0, len_exp)] * 2, [*range(len_exp, len_exp + len_ctrl)] * 2]
        filename = f"{experiment}_scores_learners.csv"
    else:
        len_exp = len(post)
        len_ctrl = len(post_ctrl)
        conds = [["exp"] * len_exp, ["exp"] * len_exp, ["control"] * len_ctrl, ["control"] * len_ctrl]
        trials = [["pretraining"] * len_exp, ["posttraining"] * len_exp, ["pretraining"] * len_ctrl,
                  ["posttraining"] * len_ctrl]
        pids = [[*range(0, len_exp)] * 2, [*range(len_exp, len_exp + len_ctrl)] * 2]
        filename = f"{experiment}_scores.csv"

    df["score"] = scores
    df["condition"] = [item for sublist in conds for item in sublist]
    df["trial"] = [item for sublist in trials for item in sublist]
    df["pid"] = [item for sublist in pids for item in sublist]
    df.to_csv(filename, index=False)


def analyze_scores(data, experiment, condition, number_of_training_trials):
    pre = data[data["block"] == "pretraining"]
    train = data[data["block"] == "training"]
    test = data[data["block"] == "test"]

    print("Pretraining mean/std:", pre["score"].mean(), pre["score"].std())
    print("Test mean/std:", test["score"].mean(), test["score"].std())

    delta = test["score"].values - pre["score"].values
    _, lower, upper = mean_confidence_interval(delta)
    print(f"95% CI for score difference: lower {lower}, upper {upper}")

    alt = "less" if experiment == "with_click" else "greater"
    test_result = wilcoxon(pre["score"], test["score"], alternative=alt)
    print("Wilcoxon test result:", test_result)

    if condition == "exp":
        learners = filter_learning_pid(train)
        train_learners = train[train["pid"].isin(learners)]
        pre_learners = pre[pre["pid"].isin(learners)]
        test_learners = test[test["pid"].isin(learners)]

        delta_learners = test_learners["score"].values - pre_learners["score"].values
        _, lower, upper = mean_confidence_interval(delta_learners)
        print(f"Learner delta CI: lower {lower}, upper {upper}")

        learner_test_result = wilcoxon(pre_learners["score"], test_learners["score"], alternative=alt)
        print("Learner Wilcoxon result:", learner_test_result)
        return pre, test, pre_learners, test_learners, learners
    return pre, test, None, None, []


def load_data(experiment, condition):
    base_path = f"../../data/human/existence_{experiment}_{condition}"
    data = pd.read_csv(os.path.join(base_path, "mouselab-mdp.csv"))
    participants = pd.read_csv(os.path.join(base_path, "participants.csv"))
    return data, len(participants)


def run_comparison_tests(experiment, condition, pre, test, other_pre, other_test, learners=None, pre_learn=None,
                         test_learn=None):
    delta_exp = test["score"].values - pre["score"].values
    delta_ctrl = other_test["score"].values - other_pre["score"].values
    alt = "greater" if experiment == "with_click" else "less"

    print(f"delta_exp mean/std: {delta_exp.mean()}, {delta_exp.std()}")
    print(f"delta_ctrl mean/std: {delta_ctrl.mean()}, {delta_ctrl.std()}")

    result = ranksums(delta_exp, delta_ctrl, alternative=alt)
    print("Ranksums test result:", result)

    # Optional: learners-specific Chi^2
    if learners and pre_learn is not None and test_learn is not None:
        if experiment == "no_click":
            count_exp = sum(pre["score"].isin([40, 43])) - sum(test["score"].isin([40, 43]))
            count_ctrl = sum(other_pre["score"].isin([40, 43])) - sum(other_test["score"].isin([40, 43]))
            chisq = chisquare([count_exp, count_ctrl])
            print("Chi^2 learners (best path, no_click):", chisq)
        else:
            inc_exp = sum(test["score"] > 6) - sum(pre["score"] > 6)
            inc_ctrl = sum(other_test["score"] > 6) - sum(other_pre["score"] > 6)
            chisq = chisquare([inc_exp, inc_ctrl])
            print("Chi^2 learners (best path, with_click):", chisq)


if __name__ == "__main__":
    experiment = "with_click"  # or "no_click"
    condition = "exp"  # or "control"
    num_trials = 30

    data, n_participants = load_data(experiment, condition)
    print("Number of participants:", n_participants)

    pre, test, pre_learn, test_learn, learners = analyze_scores(data, experiment, condition, num_trials)

    # Get data from other condition
    other_condition = "control" if condition == "exp" else "exp"
    other_data, _ = load_data(experiment, other_condition)
    other_pre = other_data[other_data["block"] == "pretraining"]
    other_test = other_data[other_data["block"] == "test"]

    run_comparison_tests(experiment, condition, pre, test, other_pre, other_test, learners, pre_learn, test_learn)

    if condition == "exp":
        train = data[data["block"] == "training"]
        train["trial_index"] = list(range(1, num_trials + 1)) * n_participants
        avg_score = train.groupby("trial_index")["score"].mean().reset_index(drop=True)
        plot_score(pre["score"], avg_score, test["score"], other_pre["score"], other_test["score"], experiment)

        # Create CSVs
        create_csv_for_r(pre, test, other_pre, other_test, experiment, learners=False)
        if pre_learn is not None and test_learn is not None:
            create_csv_for_r(pre_learn, test_learn, other_pre, other_test, experiment, learners=True)
