import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk

"""
For the best model selected through model selection, we will plot the PID behaviour vs. model behaviour
For plotting, we will let the best model with corresponding parameters (fitted using PID click sequence) interact in the mouselab (simulate)
The simulations are saved in the data pkls
"""

exp_num_list = ["v1.0",
                "c2.1",
                "c1.1",
                "high_variance_high_cost",
                "high_variance_low_cost",
                "low_variance_high_cost",
                "low_variance_low_cost"]

learning_participants = {
    "high_variance_high_cost": [32, 49, 57, 60, 94, 108, 109, 111, 129, 134, 139, 149, 161, 164, 191, 195],
    "high_variance_low_cost": [7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 145,
                               146, 154, 158, 180, 189, 197],
    "low_variance_high_cost": [2, 13, 14, 24, 28, 36, 45, 61, 62, 69, 73, 79, 80, 86, 100, 107, 124, 128, 135, 138,
                               160, 166, 171, 174, 183, 201, 206],
    "low_variance_low_cost": [9, 42, 52, 85, 110, 115, 143, 165, 172],
    "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
             82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
             160, 165, 169, 173],
    "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
             99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
             170],
    "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
             100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
             168, 171]}

# exp_num_list = ["high_variance_high_cost",
#                 "high_variance_low_cost",
#                 "low_variance_high_cost",
#                 "low_variance_low_cost"]
# exp_num_list = ["v1.0", "c2.1", "c1.1"]
# exp_num_list = ["v1.0"]
model_list = [491]

# create df with selected models
for model in model_list:
    # get score of this model in all conditions and average across then
    score_list = []
    pid_score_df = pd.DataFrame()
    for exp_name in exp_num_list:
        pid_df = pd.read_csv(f"../../data/human/{exp_name}/mouselab-mdp.csv")
        # pid_df = pid_df[pid_df["pid"].isin(learning_participants[exp_name])]
        pid_score_array = pd.pivot_table(pid_df, values="score", index="trial_index", columns="pid").T
        pid_score_df = pid_score_df.append(pid_score_array)

        # score_list is list in list, containing score of each model in a list (each n number of trials long)
        for file in os.listdir(f"results_8000_iterations/mcrl/{exp_name}_data"):  # server
            if str(model) in file:
                data = pd.read_pickle(f"results_8000_iterations/mcrl/{exp_name}_data/{file}")  # server
                score_list.append(data["r"][0])

    # get mean and sem of score_list
    model_mean_score = np.average(score_list, axis=0)
    model_sem_score = stats.sem(score_list)

    # Mann kendall test
    test_results = mk.original_test(model_mean_score)
    print(f"Mann Kendall Test for trend for {model}: {test_results}")
    # ci = 1.96 * model_sem_score
    # plt.fill_between(list(range(0, 35)), model_mean_score - ci, model_mean_score + ci, alpha=0.1,
    #                  label='Model 95% CI')

    if model == 491:
        plt.plot(model_mean_score, label="REINFORCE")
    else:
        plt.plot(model_mean_score, label="REINFORCE with pseudo-reward")

# pid_mean_score = pid_score_df.groupby("trial_index")["score"].mean().to_frame()["score"]  # output is series(pid; score)
pid_mean_score = np.mean(pid_score_df, axis=0)
test_results_pid = mk.original_test(pid_mean_score)
print(f"Mann Kendall Test for trend for pid: {test_results_pid}")
pid_sem_score = stats.sem(pid_score_df)
plt.plot(pid_mean_score, label="Participant")
ci = 1.96 * pid_sem_score
plt.fill_between(list(range(0, 35)), pid_mean_score - ci, pid_mean_score + ci, alpha=0.1,
                 label='Participant 95% CI')

# plt.ylim([-70, 40])
plt.xlabel("Trial number")
plt.ylabel("Score")
plt.legend()
plt.savefig(f"results_8000_iterations/plots/all_score_reinforce_only.png")
# plt.show()
plt.close()
