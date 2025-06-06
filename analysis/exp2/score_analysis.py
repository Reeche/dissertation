import pandas as pd
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import ast
from vars import clicking_pid, assign_model_names, participants_starting_optimally  # Added missing import


def individual_scores(exp, sampled_pid):
    """
    Plot score development for sampled individuals in the given experiment.
    """
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    for pid in sampled_pid:
        sampled_df = df[df["pid"] == pid]
        plt.plot(range(0, 35), sampled_df["score"])

    best_scores = {"v1.0": 39.99, "c2.1": 28.55, "c1.1": 6.58}
    if exp in best_scores:
        plt.axhline(y=best_scores[exp], color='b', label='Best strategy score')

    plt.legend()
    plt.title(exp)
    plt.savefig(f"plots/score/{exp}_individual_score_development.png")
    plt.close()


def lme():
    """
    Fit Linear Mixed Effects models for each experimental condition on actual score.
    """
    # Load data for all conditions
    df_inc = pd.read_csv("../../data/human/v1.0/mouselab-mdp.csv")
    df_inc['condition'] = "increasing"
    df_inc['potential_improv'] = df_inc['score'] - 39.95

    df_dec = pd.read_csv("../../data/human/c2.1/mouselab-mdp.csv")
    df_dec['condition'] = "decreasing"
    df_dec['potential_improv'] = df_dec['score'] - 28.55

    df_con = pd.read_csv("../../data/human/c1.1/mouselab-mdp.csv")
    df_con['condition'] = "constant"
    df_con['potential_improv'] = df_con['score'] - 6.57

    df = pd.concat([df_inc, df_dec, df_con], ignore_index=True)
    df = df[["pid", "trial_index", "score", "condition", "potential_improv"]]

    formula_ = "score ~ trial_index"
    for condition in ["increasing", "decreasing", "constant"]:
        temp_df = df[df['condition'] == condition]
        gamma_model = smf.mixedlm(formula=formula_, data=temp_df, groups=temp_df["pid"]).fit()
        print(f"Condition: {condition}")
        print(gamma_model.summary())


def proportion_whose_score_improved(exp):
    """
    Calculate the number of participants whose actual scores improved significantly.
    """
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    pid_list = df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        scores = df[df['pid'] == pid]["score"].to_list()
        result = mk.original_test(scores)
        if result.s > 0:
            good_pid.append(pid)
    print(f"Number of participants with improving scores in {exp}: {len(good_pid)}")


def proportion_whose_expected_strategy_score_improved(data):
    """
    Calculate the proportion of participants whose expected strategy scores improved.
    """
    improved_pid = []
    worse_pid = []
    for pid in data:
        result = mk.original_test(data[pid])
        if result.s > 0:
            improved_pid.append(pid)
        elif result.s < 0:
            worse_pid.append(pid)
    total = len(improved_pid) + len(worse_pid)
    print(f"{len(improved_pid)} out of {total} ({len(improved_pid) / total:.2f}) improved")


def remove_duplicates_from_end(lst):
    """
    Remove duplicate consecutive scores at the end of the list.
    """
    i = len(lst) - 1
    while i > 0:
        if lst[i] == lst[i - 1]:
            lst.pop(i)
        else:
            break
        i -= 1
    return lst


def proportion_of_expected_score_increase(data):
    """
    Calculate the proportion of participants whose expected scores increased once duplicates at the end are removed.
    """
    pid_dict = {col: remove_duplicates_from_end(list(data[col])) for col in data}

    increased_list = []
    for pid, scores in pid_dict.items():
        if len(scores) > 1:
            result = mk.original_test(scores)
            if result.s > 0:
                increased_list.append(pid)
    print(f"{len(increased_list)} out of {data.shape[1]} ({len(increased_list) / data.shape[1]:.2f}) have increased their expected strategy score")


def create_pairs(lst):
    """
    Create pairs of consecutive different scores from a list.
    """
    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1) if lst[i] != lst[i + 1]]


def count_of_expected_score_improvement(data, exp):
    """
    Count how many strategy pairs showed improvement in expected scores.
    """
    if exp != "c1.1":
        # Remove participant starting optimally (assuming only one)
        data = data.drop(columns=[participants_starting_optimally[exp]])

    pairs = []
    for column in data:
        scores = list(data[column])
        pairs.extend(create_pairs(scores))

    count_smaller = sum(1 for tup in pairs if tup[0] < tup[1])
    count_not_smaller = sum(1 for tup in pairs if tup[0] > tup[1])
    total = count_smaller + count_not_smaller
    print(f"Out of {total} strategy pairs, {count_smaller} ({(count_smaller / total) * 100:.2f}%) pairs showed expected score improvement.")


def potential_improvement(data):
    """
    Calculate and test for potential improvement differences across conditions.
    """
    improv_data = {}
    for exp_name, df in data.items():
        if exp_name == "v1.0":
            improv_data[exp_name] = 39.95 - df
        elif exp_name == "c2.1":
            improv_data[exp_name] = 28.55 - df
        elif exp_name == "c1.1":
            improv_data[exp_name] = 6.57 - df

    a = improv_data["v1.0"].values.flatten()
    b = improv_data["c2.1"].values.flatten()
    c = improv_data["c1.1"].values.flatten()

    S, p = stats.kruskal(a, b, c)
    print(f"Kruskal test: S={S:.4f}, p={p:.4f}")

    res = stats.mannwhitneyu(a, b, alternative="greater")
    print(f"Increasing vs decreasing (Mann-Whitney U): {res}")

    res = stats.mannwhitneyu(b, c, alternative="greater")
    print(f"Decreasing vs Constant (Mann-Whitney U): {res}")

    test = stats.ttest_ind(b, c, equal_var=False, alternative="greater")
    print(f"Decreasing vs Constant (T-test): {test}")

    return improv_data


def lme_expected_score(data):
    """
    Fit Linear Mixed Effects models on expected scores for each condition.
    """
    temp_list = []
    for exp_name, df in data.items():
        new_df = df.stack().reset_index()
        new_df.columns = ['pid', 'trial_index', 'expected_score']
        if exp_name == "v1.0":
            new_df['condition'] = "increasing"
        elif exp_name == "c2.1":
            new_df['condition'] = "decreasing"
        elif exp_name == "c1.1":
            new_df['condition'] = "constant"
        temp_list.append(new_df)

    concat_data = pd.concat(temp_list, ignore_index=True)

    formula_ = "expected_score ~ trial_index"
    for condition in ["increasing", "decreasing", "constant"]:
        temp_df = concat_data[concat_data['condition'] == condition]
        gamma_model = smf.mixedlm(formula=formula_, data=temp_df, groups=temp_df["pid"]).fit()
        print(f"Condition: {condition}")
        print(gamma_model.summary())


def distance_between_strategies(exp_list):
    """
    Statistical tests on differences between strategy scores for different experiments.
    """
    inc = pd.read_pickle("../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
    dec = pd.read_pickle("../../results/cm/strategy_scores/c2.1_strategy_scores.pkl")
    constant = pd.read_pickle("../../results/cm/strategy_scores/c1.1_strategy_scores.pkl")

    exp_scores = {"v1.0": inc, "c2.1": dec, "c1.1": constant}
    all_diffs = {}
    for exp in exp_list:
        pairs = create_pairs(list(exp_scores[exp].values()))
        differences = [tup[0] - tup[1] for tup in pairs]
        all_diffs[exp] = differences

    a = all_diffs["v1.0"]
    b = all_diffs["c2.1"]
    c = all_diffs["c1.1"]

    S, p = stats.kruskal(a, b, c)
    print(f"Kruskal test: S={S:.4f}, p={p:.4f}")

    res = stats.mannwhitneyu(a, b, alternative="greater")
    print(f"Increasing vs decreasing: {res}")

    res = stats.mannwhitneyu(c, b, alternative="less")
    print(f"Decreasing vs constant: {res}")

    res = stats.mannwhitneyu(c, a, alternative="less")
    print(f"Increasing vs constant: {res}")


def actual_score(experiment):
    """
    Analyze and plot actual scores for a given experiment.
    """
    data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
    data = data[data["pid"].isin(clicking_pid[experiment])]
    data["condition"] = experiment

    # Mann-Kendall test on scores per participant is commented out, but here's an example:
    result = mk.original_test(data["score"])
    print(f"Mann-Kendall test result for {experiment} (all scores): {result}")

    average_score = data.groupby("trial_index")["score"].mean().reset_index()
    mk_result = mk.original_test(average_score["score"])
    print(f"Mann-Kendall test result for {experiment} (average score over trials): {mk_result}")

    plt.figure(figsize=(10, 6))
    plt.plot(average_score["trial_index"], average_score["score"], marker='o', label='Average Score')
    plt.title(f"Actual Score Over Trials - {experiment}")
    plt.xlabel("Trial Index")
    plt.ylabel("Score")
    plt.show()
    plt.close()


def mer(exp):
    """
    Mixed effects regression on model MER data.
    """
    print(f"Running MER analysis for {exp}")
    data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)
    data = data[data["pid"].isin(clicking_pid[exp])]
    data = data[data["model_index"] == "491"]

    data = process_data(data, "model_mer", "pid_mer")

    data = data.explode(['pid_mer']).reset_index(drop=False)
    data["trial_index"] = data.groupby("pid_mer").cumcount() + 1

    data["pid_mer"] = data["pid_mer"].astype(float)
    data["trial_index"] = data["trial_index"].astype(float)

    formula_ = "pid_mer ~ trial_index"
    gamma_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
    print(gamma_model.summary())


def process_data(data, model_col, pid_col):
    """
    Convert string representation of lists in dataframe columns back to lists.
    """
    data[pid_col] = data[pid_col].apply(ast.literal_eval)
    data[model_col] = data[model_col].apply(ast.literal_eval)
    return data


if __name__ == "__main__":
    exp_list = ["v1.0", "c2.1", "c1.1"]

    # Run LME on actual scores
    lme()

    # Analyze distance between strategy scores for experiments
    distance_between_strategies(exp_list)

    all_data = {}
    for exp in exp_list:
        strategy_df = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{exp}_training/strategies.pkl")
        )

        # Load strategy score mapping (currently unused, but could be re-enabled)
        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/{exp}_strategy_scores.pkl")

        strategy_df = strategy_df - 1  # adjust indexing if needed

        # Filter for clicking participants
        strategy_df = strategy_df[clicking_pid[exp]]

        proportion_of_expected_score_increase(strategy_df)
        proportion_whose_expected_strategy_score_improved(strategy_df)
        proportion_whose_score_improved(exp)

        all_data[exp] = strategy_df

    # Test potential improvement differences
    improv_data = potential_improvement(all_data)

    # Fit LME on expected scores
    lme_expected_score(all_data)

    # Run actual score analysis on one experiment (example)
    for exp in exp_list:
        actual_score(exp)

