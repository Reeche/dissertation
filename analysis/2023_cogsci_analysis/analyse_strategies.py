import pandas as pd
import ast
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pymannkendall as mk

exp_num_list = ["v1.0", "c2.1", "c1.1"]
# exp_num_list = ["c1.1"]
selected_model = [491]

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

# classify the first clicks
# click_dict = {"outer_nodes": [3, 4, 7, 8, 11, 12],
#               "immediate_nodes": [1, 5, 9],
#               "middle_nodes": [2, 6, 10]}

click_dict = {3: "outer_nodes",
              4: "outer_nodes",
              7: "outer_nodes",
              8: "outer_nodes",
              11: "outer_nodes",
              12: "outer_nodes",
              1: "immediate_nodes",
              5: "immediate_nodes",
              9: "immediate_nodes",
              2: "middle_nodes",
              6: "middle_nodes",
              10: "middle_nodes",
              0: "no_click"}


model_proportion = []
pid_proportion = []
for exp_name in exp_num_list:
    for model in selected_model:
        proportion_pid = 0
        proportion_model = 0

        df = pd.read_csv(f"results_8000_iterations/{exp_name}_intermediate_results.csv")
        # filter for learners
        df = df[df["pid"].isin(learning_participants[exp_name])]
        # filter for model
        df = df[df["model"] == model]
        df['pid_clicks'] = df["pid_clicks"].apply(lambda x: ast.literal_eval(str(x)))
        df['model_clicks'] = df["model_clicks"].apply(lambda x: ast.literal_eval(str(x)))


        ### PID
        # for pid create df with trial and click sequence of each pid
        df_temp_pid = pd.DataFrame(df['pid_clicks'].to_list(),
                                   columns=['1', '2', '3', '4', '5', '6', '7', '8', '9',
                                            '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                            '19', '20', '21', '22', '23', '24', '25', '26', '27',
                                            '28', '29', '30', '31', '32', '33', '34', '35'])
        df_temp_temp_pid = df_temp_pid.copy()
        # get the proportion of adaptiveness for each trial
        # iterate through the df
        for column in df_temp_pid:
            first_click = []
            for row in df_temp_pid[column]:
                # get first click
                if len(row) > 0:
                    first_click.append(list(map(int, row))[0])
                else:  # if no clicks were made
                    first_click.append(0)
            df_temp_temp_pid[column] = first_click

        # replace the first clicks of each trial by 1 (adaptive) or 0 (not adaptive)
        df_temp_temp_pid = df_temp_temp_pid.replace(click_dict)

        if exp_name == "v1.0":
            df_temp_temp_pid = df_temp_temp_pid.replace({"outer_nodes": 1})
            df_temp_temp_pid = df_temp_temp_pid.replace({"immediate_nodes": 0})
            df_temp_temp_pid = df_temp_temp_pid.replace({"middle_nodes": 0})
            df_temp_temp_pid = df_temp_temp_pid.replace({"no_click": 0})
        elif exp_name == "c2.1":
            df_temp_temp_pid = df_temp_temp_pid.replace({"immediate_nodes": 1})
            df_temp_temp_pid = df_temp_temp_pid.replace({"outer_nodes": 0})
            df_temp_temp_pid = df_temp_temp_pid.replace({"middle_nodes": 0})
            df_temp_temp_pid = df_temp_temp_pid.replace({"no_click": 0})
        else:
            df_temp_temp_pid = df_temp_temp_pid.replace({"immediate_nodes": 1, "middle_nodes": 1})
            df_temp_temp_pid = df_temp_temp_pid.replace({"outer_nodes": 0})
            df_temp_temp_pid = df_temp_temp_pid.replace({"no_click": 0})

        # get the adaptive proportion of pid
        pid_sem_score = stats.sem(df_temp_temp_pid)

        proportion_pid = df_temp_temp_pid.sum(axis=0) / len(learning_participants[exp_name])
        pid_proportion.append(proportion_pid)

        ### MODEL
        # for pid create df with trial and click sequence of each pid
        df_temp_model = pd.DataFrame(df['model_clicks'].to_list(),
                                   columns=['1', '2', '3', '4', '5', '6', '7', '8', '9',
                                            '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                            '19', '20', '21', '22', '23', '24', '25', '26', '27',
                                            '28', '29', '30', '31', '32', '33', '34', '35'])
        df_temp_temp_model = df_temp_model.copy()
        # get the proportion of adaptiveness for each trial
        # iterate through the df
        for column in df_temp_model:
            first_click = []
            for row in df_temp_model[column]:
                # get first click
                if len(row) > 0:
                    first_click.append(list(map(int, row))[0])
                else:  # if no clicks were made
                    first_click.append(0)
            df_temp_temp_model[column] = first_click

        # replace the first clicks of each trial by 1 (adaptive) or 0 (not adaptive)
        df_temp_temp_model = df_temp_temp_model.replace(click_dict)

        if exp_name == "v1.0":
            df_temp_temp_model = df_temp_temp_model.replace({"outer_nodes": 1})
            df_temp_temp_model = df_temp_temp_model.replace({"immediate_nodes": 0})
            df_temp_temp_model = df_temp_temp_model.replace({"middle_nodes": 0})
            df_temp_temp_model = df_temp_temp_model.replace({"no_click": 0})
        elif exp_name == "c2.1":
            df_temp_temp_model = df_temp_temp_model.replace({"immediate_nodes": 1})
            df_temp_temp_model = df_temp_temp_model.replace({"outer_nodes": 0})
            df_temp_temp_model = df_temp_temp_model.replace({"middle_nodes": 0})
            df_temp_temp_model = df_temp_temp_model.replace({"no_click": 0})
        else:
            df_temp_temp_model = df_temp_temp_model.replace({"immediate_nodes": 1})
            df_temp_temp_model = df_temp_temp_model.replace({"middle_nodes": 1})
            df_temp_temp_model = df_temp_temp_model.replace({"outer_nodes": 0})
            df_temp_temp_model = df_temp_temp_model.replace({"no_click": 0})
        # get the adaptive proportion of pid
        proportion_model = df_temp_temp_model.sum(axis=0) / len(learning_participants[exp_name])
        model_proportion.append(proportion_model)

        # plt.plot(proportion_pid, label="Participant")

        # get mean and sem of score_list
        if model != 483:
            plt.plot(proportion_model, label="REINFORCE")
            test_results = mk.original_test(proportion_model)
            print(f"Mann Kendall Test for trend for {model} and {exp_name}: {test_results}")
        else:
            plt.plot(proportion_model, label="REINFORCE with pseudo-reward")
            test_results = mk.original_test(proportion_model)
            print(f"Mann Kendall Test for trend for {model} and {exp_name}: {test_results}")

    plt.plot(proportion_pid, label="Participant")
    test_results = mk.original_test(proportion_pid)
    print(f"Mann Kendall Test for trend for participant and {exp_name}: {test_results}")
    ci = 1.96 * pid_sem_score
    plt.fill_between(list(range(0, 35)), proportion_pid - ci, proportion_pid + ci, alpha=0.2,
                     label='Participant 95% CI')
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 35, 3))
    plt.xlabel("Trial number")
    plt.ylabel("Proportion of adaptive strategies")
    # plt.title(exp_name)
    plt.legend()
    plt.savefig(f"results_8000_iterations/plots/proportion_{exp_name}_reinforce_only.png")
    # plt.show()
    plt.close()

##sum of all proportions
average_model = np.mean(model_proportion, axis=0)
average_pid = np.mean(pid_proportion, axis=0)

### plot pid vs model
plt.plot(average_pid, label="Participant")

# get mean and sem of score_list
plt.plot(average_model, label="Model")

# plt.ylim([1, 8])
plt.xticks(np.arange(0, 35, 3))
plt.xlabel("Trial number")
plt.ylabel("Proportion of adaptive strategies")
# plt.title(exp_name)
plt.legend()
plt.savefig("results_8000_iterations/plots/proportion_reinforce_only.png")
# plt.show()
plt.close()