import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# exp_num_list = ["high_variance_high_cost",
#                 "high_variance_low_cost",
#                 "low_variance_high_cost",
#                 "low_variance_low_cost"]
# exp_num_list = ["v1.0", "c2.1", "c1.1"]
# model_list = ["1855", "1758"]
# model_list = [26, 27, 30, 31, 58, 59, 62, 63, 90, 91, 94, 95, 122, 123, 126, 127, 154, 155,
#               158, 159, 186, 187, 190, 191, 602, 603, 606, 607, 634, 635, 638, 639, 666, 667,
#               670, 671, 698, 699, 702, 703, 730, 731, 734, 735, 762, 763, 766, 767, 1178, 1179,
#               1182, 1183, 1210, 1211, 1214, 1215, 1242, 1243, 1246, 1247, 1274, 1275, 1278, 1279,
#               1306, 1307, 1310, 1311, 1338, 1339, 1342, 1343, 1754, 1755, 1758, 1759, 1850, 1851,
#               1854, 1855, 1946, 1947, 1950, 1951]
model_list = ["1919"]
for model in model_list:
    model_average_average_score = []
    pid_average_average_score = []
    for exp_name in exp_num_list:
        # for model in model_list:
        # for all conditions, load the score of the selected model from data pkls
        score_list = []
        for file in os.listdir(f"../results/mcrl/{exp_name}_data"):
            if str(model) in file:
                data = pd.read_pickle(f"../results/mcrl/{exp_name}_data/{file}")
                score_list.append(data["r"][0])

        # get the average of model performance
        model_score_array = np.array(score_list)
        model_average_score = np.average(model_score_array, axis=0)

        # get the average of pid performance
        pid_df = pd.read_csv(f"../data/human/{exp_name}/mouselab-mdp.csv")
        pid_score_array = np.array(pd.pivot_table(pid_df, values="score", index="trial_index", columns="pid").T)
        pid_average_score = pid_df.groupby("trial_index")["score"].mean().to_frame()[
            "score"]  # output is series(pid; score)

        # plot average across all conditions in one experiment
        model_average_average_score.append(model_average_score)
        pid_average_average_score.append(pid_average_score)

    # plot average
    model_average_average_score = np.average(model_average_average_score, axis=0)
    pid_average_average_score = np.average(pid_average_average_score, axis=0)

    plt.plot(model_average_average_score, label="Model")
    plt.plot(pid_average_average_score, label="Participant")
    # add 95% CI
    # ci = 1.96 * np.std(average_score)/np.sqrt(len(average_score))

    # get standard error of the models
    model_sem = stats.sem(model_average_average_score)
    ci = 1.96 * model_sem
    plt.fill_between(list(range(0, 35)), model_average_average_score - ci, model_average_average_score + ci, alpha=0.3,
                     label='Model 95% CI')

    # get standard error of the models
    participant_sem = stats.sem(pid_average_average_score)
    ci = 1.96 * participant_sem
    plt.fill_between(list(range(0, 35)), pid_average_average_score - ci, pid_average_average_score + ci, alpha=0.3,
                     label='Participant 95% CI')
    # plt.ylim([6, 25])
    plt.ylim([-60, 30])

    plt.xlabel("Trial number")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"results/plots/all_{model}.png")
    plt.close()
    # plt.show()
