import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# exp_num_list = ["v1.0", "c2.1", "c1.1"]
exp_num_list = ["v1.0"]

# model_list = [527, 491, 479, 1743, 1756, "mb"]
model_dict = {
    527: "RSSL",
    491: "Reinforce",
    479: "LVOC",
    1743: "Habit",
    1756: "No learning",
    "mb": "mb"
}
# model_list = ["mb"]

learning_participants_three_cond = {
    "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
             82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
             160, 165, 169, 173],
    "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
             99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
             170],
    "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
             100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
             168, 171]}

# create df with selected models
for model in model_dict.keys():
    # get score of this model in all conditions and average across then
    score_list = []
    pid_score_df = pd.DataFrame()
    for exp_name in exp_num_list:
        pid_df = pd.read_csv(f"../../data/human/{exp_name}/mouselab-mdp.csv")
        # pid_df = pid_df[pid_df["pid"].isin(learning_participants_three_cond[exp_name])]
        pid_score_array = pd.pivot_table(pid_df, values="score", index="trial_index", columns="pid").T
        pid_score_df = pid_score_df.append(pid_score_array)

        if model == 527:
            for file in os.listdir(f"../../rssl_results/mcrl/{exp_name}_data"):
                if str(model) in file:
                    data = pd.read_pickle(f"../../rssl_results/mcrl/{exp_name}_data/{file}")
                    score_list.append(data["r"][0])
        elif model == "mb":
            for file in os.listdir(f"../../results/mcrl/{exp_name}_model_based/data"):
                data = pd.read_pickle(f"../../results/mcrl/{exp_name}_model_based/data/{file}")
                score_list.append(data["rewards"][0])
        else:
            for file in os.listdir(f"../../results_400_second_fit/mcrl/{exp_name}_data"):
                if str(model) in file:
                    data = pd.read_pickle(f"../../results_400_second_fit/mcrl/{exp_name}_data/{file}")
                    score_list.append(data["r"][0])

    # get mean and sem of score_list
    model_mean_score = np.average(score_list, axis=0)
    model_sem_score = stats.sem(score_list)


    # pid_mean_score = pid_score_df.groupby("trial_index")["score"].mean().to_frame()["score"]  # output is series(pid; score)
    pid_mean_score = np.mean(pid_score_df, axis=0)
    pid_sem_score = stats.sem(pid_score_df)
    plt.plot(pid_mean_score, label="Participant")
    plt.plot(model_mean_score, label=model_dict.get(model))
    ci = 1.96 * pid_sem_score
    plt.fill_between(list(range(0, 35)), pid_mean_score - ci, pid_mean_score + ci, alpha=0.1,
                     label='Participant 95% CI')

    # plt.ylim([-70, 40])
    plt.xlabel("Trial number")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"plots/{model_dict.get(model)}.png")
    # plt.show()
    plt.close()
