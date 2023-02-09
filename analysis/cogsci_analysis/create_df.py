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


def create_dataframe_of_fitted_model_pid(exp_num, pid_list, model_list, model_information_filtered):
    # not for missing ones
    df = pd.DataFrame(
        columns=["pid", "model", "loss", "AIC", "BIC", "model_clicks", "pid_clicks", "model_score", "pid_score"])
    df["pid"] = sorted(pid_list * len(model_list))
    df["model"] = model_list * len(pid_list)

    temp_loss = []
    temp_AIC = []
    temp_BIC = []
    temp_model_clicks = []
    temp_model_score = []
    for index, row in df.iterrows():
        try:  # load loss
            # prior_data = pd.read_pickle(
            #     f"results/mcrl/{exp_num}_priors/{row['pid']}_likelihood_{row['model']}.pkl")  # local
            prior_data = pd.read_pickle(
                f"../results/mcrl/{exp_num}_priors/{row['pid']}_likelihood_{row['model']}.pkl")  # server
            losses = [trial["result"]["loss"] for trial in prior_data[0][1]]
            min_loss = min(np.absolute(losses))
            temp_loss.append(min_loss)

            # Calculate the AIC
            pr_weight_boolean = model_information_filtered.filter(items=[row['model']], axis=0)["pr_weight"].values
            # pr_weight is always added regardless of pr, therefore one more (unused) parameter need to be deducated
            if pr_weight_boolean[0]:  # if pr_weight is true
                number_of_parameters = len(prior_data[1]) - 56  # there are 56 strategy weight features
            else: #because pr_weight is always something added even if not used
                number_of_parameters = len(prior_data[1]) - 57

            # min_loss is log-likelihood
            AIC = 2 * min_loss + number_of_parameters * 2  # previous aic by Yash
            temp_AIC.append(AIC)

            BIC = 2 * min_loss + number_of_parameters * np.log(35)
            temp_BIC.append(BIC)

        except Exception as e:
            temp_loss.append("na")
            temp_AIC.append("na")
            temp_BIC.append("na")
            print(f"PID {row['pid']} and model {row['model']} priors pickle did not work: Error: {e}")

        try:  # load click and score data of model
            # reward_data = pd.read_pickle(f"results/mcrl/{exp_num}_data/{row['pid']}_{row['model']}_1.pkl")  # local
            reward_data = pd.read_pickle(f"../results/mcrl/{exp_num}_data/{row['pid']}_{row['model']}_1.pkl")  # server
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
    # pid_info = pd.read_csv(f"data/human/{exp_num}/mouselab-mdp.csv")  # local
    pid_info = pd.read_csv(f"../data/human/{exp_num}/mouselab-mdp.csv")  # server
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
    df.to_csv(f"results/{exp_num}_intermediate_results_v4.csv")  # server

    return df


def create_bic_table(df, exp_num):
    # creates a bic table that is used for Bayesian Model Comparison in Matlab
    # df = df.drop(columns=['loss', 'AIC', 'model_clicks', 'pid_clicks', 'model_score', 'pid_score'])
    # df = df.reset_index()
    # aggfunc is needed, sum should not do anything since tehre is only one BIC value
    df_pivot = pd.pivot_table(df, index=["pid"], columns=["model"], values=["BIC"], aggfunc=np.sum, fill_value=0)
    # df_pivot.to_csv(f"cogsci_analysis/results/matlab_{exp_num}_bic_filtered_for_models.csv")  # local
    df_pivot.to_csv(f"results/matlab_{exp_num}_bic_v4.csv")  # server

    return df_pivot


def create_one_bic_table():
    exp_num_list = ["c2.1",
                    "c1.1",
                    "high_variance_high_cost",
                    "high_variance_low_cost",
                    "low_variance_high_cost",
                    "low_variance_low_cost"]
    df = pd.read_csv(f"cogsci_analysis/results/matlab_v1.0_bic_v4.csv")

    for exp_name in exp_num_list:
        temp_df = pd.read_csv(f"cogsci_analysis/results/matlab_{exp_name}_bic_v4.csv")
        temp_df = temp_df.iloc[2:]
        df = df.append(temp_df)
    df.to_csv("cogsci_analysis/results/bic_all_v4.csv")
    return df


if __name__ == "__main__":
    # create_one_bic_table()
    exp_name = sys.argv[1]
    # exp_name = "v1.0"

    model_list = [27, 31, 59, 63, 91, 95, 123, 127, 155, 159, 187, 191, 411,
                  415, 443, 447, 475, 479, 507, 511, 539, 543, 571, 575, 603,
                  607, 635, 639, 667, 671, 699, 703, 731, 735, 763, 767, 987,
                  991, 1019, 1023, 1051, 1055, 1083, 1087, 1115, 1119, 1147,
                  1151, 1179, 1183, 1211, 1215, 1243, 1247, 1275, 1279, 1307,
                  1311, 1339, 1343, 1563, 1567, 1595, 1599, 1627, 1631, 1659,
                  1663, 1691, 1695, 1723, 1727, 1755, 1759, 1819, 1823, 1851,
                  1855, 1915, 1918, 1919, 1947, 1951, 2011, 2015, 5134]

    optimization_criterion = "likelihood"
    pid_list = get_all_pid_for_env(exp_name)
    # pid_list = [1, 5]
    # model_list = [769, 796]
    #
    # load model.csv
    # model_information = pd.read_csv("mcl_toolbox/models/rl_models.csv") #local
    model_information = pd.read_csv("../../mcl_toolbox/models/rl_models.csv")  # server
    # model_information = model_information[model_information[index].isin(model_list)]
    model_information_filtered = model_information.filter(items=model_list, axis=0)
    df = create_dataframe_of_fitted_model_pid(exp_name, pid_list, model_list, model_information_filtered)
    bic_table = create_bic_table(df, exp_name)
