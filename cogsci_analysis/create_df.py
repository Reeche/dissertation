import os
import sys
import pandas as pd
import ast
import numpy as np

from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env

"""
This script contains code to create intermediate df/csv that can be used for further analysis 
This script also creates a csv that is needed for MATLAB to calculate the BIC
"""


def create_dataframe_of_fitted_model_pid(exp_num, pid_list, model_list):
    # missing ones
    # if exp_num == "high_variance_high_cost":
    #     data = [[0, 1]]
    # elif exp_num == "low_variance_high_cost":
    #     data = [[13, 1358], [13, 1382], [16, 1405], [21, 1345], [21, 1354], [31, 1192], [31, 1355],
    #             [43, 1351], [80, 1388], [80, 1397], [100, 1355], [157, 1458], [160, 1357], [163, 1351],
    #             [201, 1393]]
    # elif exp_num == "c1.1":
    #     data = [[100, 1]]
    # elif exp_num == "c2.1":
    #     data = [[0, 1], [13, 1483], [20, 1893], [103, 81], [103, 1692], [108, 1233], [113, 311],
    #             [113, 529], [115, 875], [128, 946], [128, 1551], [130, 1530], [142, 1855], [149, 982],
    #             [149, 987], [164, 842]]
    # elif exp_num == "v1.0":
    #     data = [[6, 1782], [66, 1801], [68, 1970], [68, 1977], [68, 2000], [75, 870],
    #            [77, 1687], [80, 146], [82, 1366], [82, 1390], [94, 992], [98, 433]]
    # else:
    #     print("nothing missing")

    # create df with pid and model index
    # for missing onnes
    # df = pd.DataFrame(data, columns=["pid", "model"])

    # not for missing ones
    df = pd.DataFrame(columns=["pid", "model", "loss", "AIC", "BIC", "model_clicks", "pid_clicks", "model_score", "pid_score"])
    df["pid"] = sorted(pid_list * len(model_list))
    df["model"] = model_list * len(pid_list)

    temp_loss = []
    temp_AIC = []
    temp_BIC = []
    temp_model_clicks = []
    temp_model_score = []
    for index, row in df.iterrows():
        # print("For PID and model: ", row['pid'], row['model'])
        # look up loss for pid and model combination
        try:  # load loss
            prior_data = pd.read_pickle(f"results/mcrl/{exp_num}_priors/{row['pid']}_likelihood_{row['model']}.pkl") #local
            # prior_data = pd.read_pickle(f"../results/mcrl/{exp_num}_priors/{row['pid']}_likelihood_{row['model']}.pkl") #server
            losses = [trial["result"]["loss"] for trial in prior_data[0][1]]
            min_loss = min(np.absolute(losses))
            temp_loss.append(min_loss)

            # Calculate the AIC
            number_of_parameters = len(prior_data[1]) - 56 #there are 56 strategy weight features

            # min_loss is log-likelihood
            AIC = 2 * min_loss + number_of_parameters * 2 #previous aic by Yash
            temp_AIC.append(AIC)

            BIC = 2 * min_loss + number_of_parameters * np.log(35)
            temp_BIC.append(BIC)

        except Exception as e:
            temp_loss.append("na")
            temp_AIC.append("na")
            temp_BIC.append("na")
            print(f"PID {row['pid']} and model {row['model']} priors pickle did not work: Error: {e}")

        try:  # load click and score data of model
            reward_data = pd.read_pickle(f"results/mcrl/{exp_num}_data/{row['pid']}_{row['model']}_1.pkl") #local
            # reward_data = pd.read_pickle(f"../results/mcrl/{exp_num}_data/{row['pid']}_{row['model']}_1.pkl") #server
            temp_model_clicks.append(reward_data["a"][0])
            temp_model_score.append(reward_data["r"][0])

        except Exception as e:
            temp_model_score.append("na")
            temp_model_clicks.append("na")
            print(f"PID {row['pid']} and model {row['model']} data pickle did not work: Error: {e}")

    df["loss"] = temp_loss
    df["AIC"] = temp_AIC
    df["BIC"] = temp_BIC
    df["model_clicks"] = temp_model_clicks
    df["model_score"] = temp_model_score

    df["pid_clicks"] = "na"
    df["pid_score"] = "na"
    pid_info = pd.read_csv(f"data/human/{exp_num}/mouselab-mdp.csv") #local
    # pid_info = pd.read_csv(f"../data/human/{exp_num}/mouselab-mdp.csv") #server
    for pid in df["pid"]:
        pid_info_temp = pid_info.loc[pid_info['pid'] == pid]
        temp_reward_list = []
        temp_click_list = []  # contains cicks of all 35 trials
        # add pid information on score and clicks
        for index, row in pid_info_temp.iterrows():
            test = row["queries"]
            test2 = ast.literal_eval(test)
            clicks = test2["click"]["state"]["target"]
            temp_click_list.append(clicks)
            temp_reward_list.append(row["score"])
        filter_for_pid = df.loc[df['pid'] == pid]
        for idx_, row_ in filter_for_pid.iterrows():
            df.at[idx_, 'pid_clicks'] = temp_click_list
            df.at[idx_, 'pid_score'] = temp_reward_list

    # df.to_csv(f"cogsci_analysis/results/{exp_num}_intermediate_results.csv") #local
    # df.to_csv(f"results/{exp_num}_intermediate_results_{model_batch}.csv") #server

    return df


def create_bic_table(df, exp_num):
    # creates a bic table that is used for Bayesian Model Comparison in Matlab
    # df = df.drop(columns=['loss', 'AIC', 'model_clicks', 'pid_clicks', 'model_score', 'pid_score'])
    # df = df.reset_index()
    # aggfunc is needed, sum should not do anything since tehre is only one BIC value
    df_pivot = pd.pivot_table(df, index=["pid"], columns=["model"], values=["BIC"], aggfunc=np.sum, fill_value=0)
    df_pivot.to_csv(f"cogsci_analysis/results/matlab_{exp_num}_bic_filtered_for_models.csv") #local
    # df_pivot.to_csv(f"results/matlab_{exp_num}_bic_v2.csv") #server

    return df_pivot


if __name__ == "__main__":
    exp_name = sys.argv[1]
    # model_list = sys.argv[2]
    # exp_name = "v1.0"
    # model_list = list(range(0, 2016))
    model_list = [26, 27, 30, 31, 58, 59, 62, 63, 90, 91, 94, 95, 122, 123, 126, 127, 154,
                  155, 158, 159, 186, 187, 190, 191, 218, 219, 222, 223, 250, 251, 254, 255,
                  282, 283, 286, 287, 314, 315, 318, 319, 346, 347, 350, 351, 378, 379, 382,
                  383, 410, 411, 414, 415, 442, 443, 446, 447, 474, 475, 478, 479, 506, 507,
                  510, 511, 538, 539, 542, 543, 570, 571, 574, 575, 602, 603, 606, 607, 634,
                  635, 638, 639, 666, 667, 670, 671, 698, 699, 702, 703, 730, 731, 734, 735,
                  762, 763, 766, 767, 794, 795, 798, 799, 826, 827, 830, 831, 858, 859, 862,
                  863, 890, 891, 894, 895, 922, 923, 926, 927, 954, 955, 958, 959, 986, 987,
                  990, 991, 1018, 1019, 1022, 1023, 1050, 1051, 1054, 1055, 1082, 1083, 1086,
                  1087, 1114, 1115, 1118, 1119, 1146, 1147, 1150, 1151, 1178, 1179, 1182, 1183,
                  1210, 1211, 1214, 1215, 1242, 1243, 1246, 1247, 1274, 1275, 1278, 1279, 1306,
                  1307, 1310, 1311, 1338, 1339, 1342, 1343, 1370, 1371, 1374, 1375, 1402, 1403,
                  1406, 1407, 1434, 1435, 1438, 1439, 1466, 1467, 1470, 1471, 1498, 1499, 1502,
                  1503, 1530, 1531, 1534, 1535, 1562, 1563, 1566, 1567, 1594, 1595, 1598, 1599,
                  1626, 1627, 1630, 1631, 1658, 1659, 1662, 1663, 1690, 1691, 1694, 1695, 1722,
                  1723, 1726, 1727, 1754, 1755, 1758, 1759, 1786, 1787, 1790, 1791, 1818, 1819,
                  1822, 1823, 1850, 1851, 1854, 1855, 1882, 1883, 1886, 1887, 1914, 1915, 1918,
                  1919, 1946, 1947, 1950, 1951, 1978, 1979, 1982, 1983, 2010, 2011, 2014, 2015]

    optimization_criterion = "likelihood"
    pid_list = get_all_pid_for_env(exp_name)
    # model_list = [1412]
    # pid_list = [68]
    df = create_dataframe_of_fitted_model_pid(exp_name, pid_list, model_list)



    # exp_num_list = ["v1.0",
    #                 "c2.1",
    #                 "c1.1",
    #                 "high_variance_high_cost",
    #                 "high_variance_low_cost",
    #                 "low_variance_high_cost",
    #                 "low_variance_low_cost"]

    # for exp_name in exp_num_list:
    #     df = pd.read_csv(f"cogsci_analysis/results/{exp_name}_intermediate_results.csv")
    #     df = df[df['model'].isin(models_to_be_considered)]
    bic_table = create_bic_table(df, exp_name)
