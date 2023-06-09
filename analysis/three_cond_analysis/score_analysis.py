import pandas as pd
import numpy as np
from random import sample
import pymannkendall as mk
import matplotlib.pyplot as plt

exp = "v1.0"


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


def score_according_to_strategy_used(exp):
    df = pd.DataFrame.from_dict(pd.read_pickle(f"../../results/cm/inferred_strategies/{exp}_training/strategies.pkl"))

    # load strategy score mapping
    score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/{exp}_strategy_scores.pkl")

    # score_mapping start from 0 but inferred_strategies.pkl start from 1
    df = df - 1

    # replace strategy with score
    df = df.replace(score_mapping)

    # sampled_pid = sample_from_learning_pid(exp)
    ## filter df for learning participants
    # df = df[df.columns.intersection(sampled_pid)]

    pid_list = [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
                82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
                160, 165, 169, 173]

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


# score_according_to_strategy_used(exp)


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

proportion_whose_score_improved(exp)