import pandas as pd
import numpy as np
from random import sample
import pymannkendall as mk
import matplotlib.pyplot as plt



def sample_from_learning_pid(exp, n=5):
    learning_participants = {
        "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
                 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
                 160, 165, 169, 173],
        "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
                 99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
                 170],
        "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
                 100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
                 168, 171]}

    learning_pid_for_exp = learning_participants[exp]
    sampled_pid = sample(learning_pid_for_exp, n)
    return sampled_pid


def score_according_to_strategy_used(data):
    # create batches from pid_list
    # sublists = np.array_split(pid_list, 6)

    plt.plot(range(0, 35), df)
    # if exp == "v1.0":
    #     plt.axhline(y=39.99, color='r', label='Best strategy score')
    # elif exp == "c2.1":
    #     plt.axhline(y=28.55, color='r', label='Best strategy score')
    # elif exp == "c1.1":
    #     plt.axhline(y=6.58, color='r', label='Best strategy score')

    plt.legend()
    # plt.title(exp)
    plt.savefig(f"plots/score/{exp}_individual_strategy_score_all.png")
    # plt.show()
    plt.close()
    return None


# score_according_to_strategy_used(data)


def individual_scores(exp):
    # plot score development for individuals
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    sampled_pid = sample_from_learning_pid(exp)

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


def proportion_whose_score_improved(exp):
    df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    df = df[["pid", "trial_index", "score"]]

    pid_list = df["pid"].unique()
    good_pid = []
    for pid in pid_list:
        temp_list = df[df['pid'] == pid]["score"].to_list()
        result = mk.original_test(temp_list)
        if result[0] == "increasing":
            good_pid.append(pid)
    print(len(good_pid))


# proportion_whose_score_improved(exp)

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
    print(f"{len(increased_list)} out of {data.shape[1]} {len(increased_list) / data.shape[1]} have increased their expected strategy score")
    return None

def create_pairs(lst):
    pairs = []
    for i in range(len(lst)-1):
        if lst[i] != lst[i+1]:
            pair = (lst[i], lst[i+1])
            pairs.append(pair)
    return pairs

def proportion_of_expected_score_improvement(data):
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
        elif tup[0] >= tup[1]:
            count_not_smaller += 1
    print(f"out of {count_smaller + count_not_smaller} strategy pairs, "
          f"{count_smaller} ({count_smaller / (count_smaller + count_not_smaller) * 100}) pairs expected score improved.")


if __name__ == "__main__":
    learning_participants = {
        "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
                 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
                 160, 165, 169, 173],
        "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
                 99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
                 170],
        "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
                 100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
                 168, 171]}
    exp_list = ["v1.0", "c2.1", "c1.1"]
    # exp = "c1.1"
    for exp in exp_list:
        df = pd.DataFrame.from_dict(pd.read_pickle(f"../../results/cm/inferred_strategies/{exp}_training/strategies.pkl"))

        # load strategy score mapping
        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/{exp}_strategy_scores.pkl")
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        df = df - 1

        # replace strategy with score
        df = df.replace(score_mapping)

        # sampled_pid = sample_from_learning_pid(exp)
        ## filter df for learning participants
        # df = df[df.columns.intersection(learning_participants[exp])]

        # proportion_of_expected_score_increase(df)
        proportion_of_expected_score_improvement(df)

