import pandas as pd
import numpy as np
from random import sample
import pymannkendall as mk
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from vars import clicking_pid


def individual_scores(exp):
    # plot score development for individuals
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    # plot for each individual individually and add them to the plot
    for pid in sampled_pid:
        sampled_df = df[df["pid"] == pid]
        plt.plot(range(0, 35), sampled_df["score"])

    if exp == "v1.0":
        plt.axhline(y=39.99, color='b', label='Best strategy score')
    elif exp == "c2.1":
        plt.axhline(y=28.55, color='b', label='Best strategy score')
    elif exp == "c1.1":
        plt.axhline(y=6.58, color='b', label='Best strategy score')

    plt.legend()
    plt.title(exp)
    plt.savefig(f"plots/score/{exp}_individual_score_development.png")
    # plt.show()
    plt.close()


def lme():
    df_inc = pd.read_csv(f"../../data/human/v1.0/mouselab-mdp.csv")
    df_inc['condition'] = ["increasing"] * df_inc.shape[0]
    df_inc['potential_improv'] = df_inc['score'] - 39.95

    df_dec = pd.read_csv(f"../../data/human/c2.1/mouselab-mdp.csv")
    df_dec['condition'] = ["decreasing"] * df_dec.shape[0]
    df_inc['potential_improv'] = df_inc['score'] - 28.55

    df_con = pd.read_csv(f"../../data/human/c1.1/mouselab-mdp.csv")
    df_con['condition'] = ["constant"] * df_con.shape[0]
    df_inc['potential_improv'] = df_inc['score'] - 6.57

    df = pd.concat([df_inc, df_dec, df_con], ignore_index=True)
    df = df[["pid", "trial_index", "score", "condition", "potential_improv"]]
    # formula_ = "score ~ trial_index*C(condition)"
    # formula_ = "score ~ trial_index:C(condition)"
    formula_ = "score ~ trial_index"

    for condition in ["increasing", "decreasing", "constant"]:
        temp_df = df[df['condition'] == condition]
        gamma_model = smf.mixedlm(formula=formula_, data=temp_df, groups=temp_df["pid"]).fit()
        print(condition, gamma_model.summary())


    # gamma_model = smf.mixedlm(formula=formula_, data=df, groups=df["pid"]).fit()
    # print(gamma_model.summary())
    return None


def proportion_whose_score_improved(exp):
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    pid_list = df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = df[df['pid'] == pid]["score"].to_list()
        result = mk.original_test(temp_list)
        # if result[0] == "increasing":
        #     good_pid.append(pid)
        if result.s > 0:
            good_pid.append(pid)
    print(len(good_pid))


def proportion_whose_expected_strategy_score_improved(data):
    improved_pid = []
    worse_pid = []
    for pid in data:
        result = mk.original_test(data[pid])
        if result.s > 0:
            improved_pid.append(pid)
        if result.s < 0:
            worse_pid.append(pid)
    print(
        f"{len(improved_pid)}, out of {len(worse_pid) + len(improved_pid)} ({len(improved_pid) / data.shape[1]}) improved")
    return None


def remove_duplicates_from_end(lst):
    i = len(lst) - 1
    while i > 0:
        if lst[i] == lst[i - 1]:
            lst.pop(i)
        else:
            break
        i -= 1
    return lst


def proportion_of_expected_score_increase(data):
    # once they converge, the score will be counted only once
    pid_dict = {}
    for column in data:
        pid_dict[column] = remove_duplicates_from_end(list(data[column]))

    increased_list = []
    for pid, scores in pid_dict.items():
        if len(scores) > 1:
            result = mk.original_test(scores)
            if result[5] > 0:
                increased_list.append(pid)
    print(
        f"{len(increased_list)} out of {data.shape[1]} {len(increased_list) / data.shape[1]} have increased their expected strategy score")
    return None


def create_pairs(lst):
    pairs = []
    for i in range(len(lst) - 1):
        if lst[i] != lst[i + 1]:
            pair = (lst[i], lst[i + 1])
            pairs.append(pair)
    return pairs


def count_of_expected_score_improvement(data, exp):
    # remove those 2 pid from the dataframe
    if exp != "c1.1":
        data = data.drop(columns=[participants_starting_optimally[exp]])
    pairs = []
    for column in data:
        score = list(data[column])
        for i in range(len(score) - 1):
            if score[i] != score[i + 1]:
                pair = (score[i], score[i + 1])
                pairs.append(pair)

    count_smaller = 0
    count_not_smaller = 0

    for tup in pairs:
        if tup[0] < tup[1]:
            count_smaller += 1
        elif tup[0] > tup[1]:
            count_not_smaller += 1
    print(f"out of {count_smaller + count_not_smaller} strategy pairs, "
          f"{count_smaller} ({count_smaller / (count_smaller + count_not_smaller) * 100}) pairs expected score improved.")


def potential_improvement(data):
    ### calculate room for improvement
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
    print("Kruskal test", S, p)

    ## testing for pairs
    res = stats.mannwhitneyu(a, b, alternative="greater")
    print("Increasing vs decreasing: ", res)

    res = stats.mannwhitneyu(b, c, alternative="greater")
    print("Decreasing vs Constant: ", res)

    test = stats.ttest_ind(b, c, equal_var=False, alternative="greater")
    print("Decreasing vs Constant: ", test)

    return improv_data


def lme_expected_score(data):
    temp_list = []
    for exp_name, df in data.items():
        new_df = df.stack().reset_index()
        new_df.columns = ['pid', 'trial_index', 'expected_score']
        if exp_name == "v1.0":
            new_df['condition'] = "increasing"
            # new_df['potential_improvement'] = 39.95 - new_df['expected_score']
        elif exp_name == "c2.1":
            new_df['condition'] = "decreasing"
            # new_df['potential_improvement'] = 28.55 - new_df['expected_score']
        elif exp_name == "c1.1":
            new_df['condition'] = "constant"
            # new_df['potential_improvement'] = 6.57 - new_df['expected_score']
        temp_list.append(new_df)

    concat_data = pd.concat(temp_list, ignore_index=True)

    # formula_ = "expected_score ~ trial_index*C(condition)"  # adding improvement does not work
    formula_ = "expected_score ~ trial_index"  # adding improvement does not work
    for condition in ["increasing", "decreasing", "constant"]:
        temp_df = concat_data[concat_data['condition'] == condition]
        gamma_model = smf.mixedlm(formula=formula_, data=temp_df, groups=temp_df["pid"]).fit()
        print(condition, gamma_model.summary())
    # gamma_model = smf.mixedlm(formula=formula_, data=concat_data).fit()
    # gamma_model = smf.mixedlm(formula=formula_, data=concat_data, groups=concat_data["pid"]).fit()
    # print(gamma_model.summary())
    return None


### Check how much improvement there is between the expected strategy scores
def distance_between_strategies(exp_list):
    inc = pd.read_pickle(f"../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
    dec = pd.read_pickle(f"../../results/cm/strategy_scores/c2.1_strategy_scores.pkl")
    constant = pd.read_pickle(f"../../results/cm/strategy_scores/c1.1_strategy_scores.pkl")

    exp_scores = [inc, dec, constant]
    all = {}
    for exp in zip(exp_list, exp_scores):
        pairs = create_pairs(list(exp[1].values()))
        difference = []
        for tup in pairs:
            difference.append(tup[0] - tup[1])

        all[exp[0]] = difference

    a = all["v1.0"]
    b = all["c2.1"]
    c = all["c1.1"]

    ### statistcal test that the values are diffent and above 0
    S, p = stats.kruskal(a, b, c)
    print("Kruskal test", S, p)

    ## testing for pairs
    res = stats.mannwhitneyu(a, b, alternative="greater")
    print("Increasing vs decreasing: ", res)

    res = stats.mannwhitneyu(c, b, alternative="less")
    print("Decreasing vs constant: ", res)

    res = stats.mannwhitneyu(c, a, alternative="less")
    print("Increasing vs constant: ", res)

    return None

if __name__ == "__main__":
    # participants_starting_optimally = {
    #     "v1.0": [85, 140],
    #     "c2.1": [0, 8, 39, 41, 58, 72, 99, 130, 152, 172],
    #     "c1.1": [2, 7, 36, 37, 42, 89, 91, 92, 157, 168]
    # }

    exp_list = ["v1.0", "c2.1", "c1.1"]
    # lme()
    # distance_between_strategies(exp_list)

    all_data = {}

    for exp in exp_list:
        strategy_df = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{exp}_training/strategies.pkl"))

        # load strategy score mapping
        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/{exp}_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_df = strategy_df - 1

        # replace strategy with score
        # strategy_df = strategy_df.replace(score_mapping)

        ### filter for clicking participants
        strategy_df = strategy_df[clicking_pid[exp]]

        ### count_of_expected_score_improvement(strategy_df, exp)
        # proportion_of_expected_score_increase(strategy_df)

        ### Proportion of participants whose expected score increased
        # proportion_whose_expected_strategy_score_improved(strategy_df)

        ### Proportion whose score improved
        # proportion_whose_score_improved(exp)
        all_data[exp] = strategy_df



    ### test if potential improvement is different across conditions
    # improv_data = potential_improvement(all_data)
    lme_expected_score(all_data)
