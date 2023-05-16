import pandas as pd
import ast
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pymannkendall as mk
"""
Does our model perform the same amount of clicks as the participants? 
"""

# import data
# exp_num_list = ["v1.0",
#                 "c2.1",
#                 "c1.1",
#                 "high_variance_high_cost",
#                 "high_variance_low_cost",
#                 "low_variance_high_cost",
#                 "low_variance_low_cost"]
#
# model_list = [27, 31, 59, 63, 91, 95, 123, 127, 155, 159, 187, 191, 411,
#               415, 443, 447, 475, 479, 507, 511, 539, 543, 571, 575, 603,
#               607, 635, 639, 667, 671, 699, 703, 731, 735, 763, 767, 987,
#               991, 1019, 1023, 1051, 1055, 1083, 1087, 1115, 1119, 1147,
#               1151, 1179, 1183, 1211, 1215, 1243, 1247, 1275, 1279, 1307,
#               1311, 1339, 1343, 1563, 1567, 1595, 1599, 1627, 1631, 1659,
#               1663, 1691, 1695, 1723, 1727, 1755, 1759, 1819, 1823, 1851,
#               1855, 1915, 1918, 1919, 1947, 1951, 2011, 2015, 5134]

exp_num_list = ["low_variance_high_cost", "low_variance_low_cost"]
# exp_num_list = ["high_variance_high_cost", "high_variance_low_cost"]
# exp_num_list = ["v1.0", "c2.1", "c1.1"]
# exp_num_list = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
selected_model = [491]

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

for model in selected_model:
    model_df = pd.DataFrame()
    pid_df = pd.DataFrame()
    # create df with number of clicks of the participants
    for exp_name in exp_num_list:
        df = pd.read_csv(f"results_8000_iterations/{exp_name}_intermediate_results.csv")
        df_pid_clicks = df[['pid', 'pid_clicks']].copy()
        df_pid_clicks = df_pid_clicks.drop_duplicates()

        # filter for learning pid
        df_pid_clicks = df_pid_clicks[df_pid_clicks["pid"].isin(learning_participants[exp_name])]
        df_pid_clicks['pid_clicks'] = df_pid_clicks["pid_clicks"].apply(lambda x: ast.literal_eval(str(x)))

        ### get number of clicks for the participants
        number_of_clicks_pid = []
        for index, row in df_pid_clicks.iterrows():
            temp_list = []
            for trial in row["pid_clicks"]:
                temp_list.append(len(trial))
            number_of_clicks_pid.append(temp_list)
        df_pid_clicks['number_of_clicks'] = number_of_clicks_pid
        pid_df = pid_df.append(df_pid_clicks)

        ### get number of clicks of the models
        # filter for the selected model
        df_model = df[df["model"] == model]
        df_model_clicks = df_model[['model', 'model_clicks']].copy()
        df_model_clicks['model_clicks'] = df_model_clicks["model_clicks"].apply(lambda x: ast.literal_eval(str(x)))

        number_of_clicks_model = []
        for index, row in df_model_clicks.iterrows():
            temp_list_2 = []
            for trial in row["model_clicks"]:
                # remove 0 from the list
                X = [i for i in trial if i != 0]
                temp_list_2.append(len(X))
            number_of_clicks_model.append(temp_list_2)
        df_model_clicks['number_of_clicks'] = number_of_clicks_model
        model_df = model_df.append(df_model_clicks)

    # average across the number of clicks of participants
    # split the series into 35 trials
    df_temp_pid = pd.DataFrame(pid_df['number_of_clicks'].to_list(),
                               columns=['1', '2', '3', '4', '5', '6', '7', '8', '9',
                                        '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                        '19', '20', '21', '22', '23', '24', '25', '26', '27',
                                        '28', '29', '30', '31', '32', '33', '34', '35'])

    df_temp_model = pd.DataFrame(model_df['number_of_clicks'].to_list(),
                                 columns=['1', '2', '3', '4', '5', '6', '7', '8', '9',
                                          '10', '11', '12', '13', '14', '15', '16', '17', '18',
                                          '19', '20', '21', '22', '23', '24', '25', '26', '27',
                                          '28', '29', '30', '31', '32', '33', '34', '35'])
    participant_mean = df_temp_pid.mean(axis=0)
    model_mean = df_temp_model.mean(axis=0)



    # get mean and sem of score_list
    # model_sem_score = stats.sem(df_temp_model)
    if model == 483:
        plt.plot(model_mean, label="REINFORCE with pseudo-reward")
        test_results = mk.original_test(model_mean)
        print(f"Mann Kendall Test for trend for {model}: {test_results}")
    else:
        plt.plot(model_mean, label="REINFORCE")
        test_results = mk.original_test(model_mean)
        print(f"Mann Kendall Test for trend for {model}: {test_results}")


### plot pid vs model
# pid_sem_score = stats.sem(df_temp_pid)
# plt.plot(participant_mean, label="Participant")
# ci = 1.96 * pid_sem_score
# plt.fill_between(list(range(0, 35)), participant_mean - ci, participant_mean + ci, alpha=0.2,
#                  label='Participant 95% CI')
#
# # plt.ylim([1, 8])
# plt.xticks(np.arange(0, 35, 3))
# plt.xlabel("Trial number")
# plt.ylabel("Number of clicks")
# # plt.axhline(y=6.71, color='r', linestyle='-', label='Optimal amount of clicks') #high
# # plt.axhline(y=2.91, color='r', linestyle='-', label='Optimal amount of clicks') #low
# plt.legend()
# plt.savefig(f"results_8000_iterations/plots/planningamount_high_click_reinforce_only.png")
#
# # plt.show()
# plt.close()

# 7.10 (high variance, low cost), 6.32 (high variance, high cost), 5.82 (low variance, low cost), and 0 (low variance, high cost),
