import pandas as pd
from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env
from sklearn.cluster import KMeans
import numpy as np
import pymannkendall as mk


def loss(exp_name, df, pid_list):
    best_model_count = {}
    for pid in pid_list:
        # filter for that pid
        data_for_pid = df[df["pid"] == pid]
        # get the best model for that pid
        idx_lowest = pd.to_numeric(data_for_pid['loss']).idxmin()  # equals model index

        # filter for best model, not needed here because the models follow number 0 - 2016
        best_model = data_for_pid[data_for_pid.index == idx_lowest]
        best_model_name = best_model["model"].values[0]
        if best_model_name in best_model_count:
            best_model_count[best_model_name] += 1
        else:
            best_model_count[best_model_name] = 1
    print(f"############################## {exp_name} ##############################")
    print("Best model count according to AIC", best_model_count)

    ## sort by AIC
    df = df.sort_values(by=["loss"])
    df["loss"] = df["loss"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    print("Grouped model and AIC")
    sorted_df = grouped_df.sort_values(by=["loss"])
    print(sorted_df)


def aic(exp_name, df, pid_list):
    # Get best model for each participant and count which one occured most often based on AIC
    # df = df.dropna(subset=["loss", "AIC"])
    # df = df[df["AIC"].str.contains("na") == False]
    best_model_count = {}
    for pid in pid_list:
        # filter for that pid
        data_for_pid = df[df["pid"] == pid]
        # get the best model for that pid
        idx_lowest = pd.to_numeric(data_for_pid['AIC']).idxmin()  # equals model index

        # filter for best model, not needed here because the models follow number 0 - 2016
        best_model = data_for_pid[data_for_pid.index == idx_lowest]
        best_model_name = best_model["model"].values[0]
        if best_model_name in best_model_count:
            best_model_count[best_model_name] += 1
        else:
            best_model_count[best_model_name] = 1
    print(f"############################## {exp_name} ##############################")
    print("Best model count according to AIC", best_model_count)

    ## sort by AIC
    df = df.sort_values(by=["AIC"])
    df["AIC"] = df["AIC"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    print("Grouped model and AIC")
    sorted_df = grouped_df.sort_values(by=["AIC"])
    print(sorted_df)


def bic(exp_name, df_learning, pid_list):
    ## Get best model for each participant and count which one occured most often based on BIC

    # filter by pid list
    df = df_learning[df_learning["pid"].isin(pid_list)]

    ##create bic table for matlab
    df_pivot = pd.pivot_table(df_learning, index=["pid"], columns=["model"], values=["BIC"], aggfunc=np.sum,
                              fill_value=0)
    df_pivot.to_csv(f"results_400_second_fit/matlab_bic_{exp_name}_learners.csv")

    best_model_count = {}
    for pid in pid_list:
        data_for_pid = df_learning[df_learning["pid"] == pid]
        # idx_lowest = data_for_pid["BIC"].idxmin()
        idx_lowest = pd.to_numeric(data_for_pid['BIC']).idxmin()
        best_model = data_for_pid[data_for_pid.index == idx_lowest]
        best_model_name = best_model["model"].values[0]
        if best_model_name in best_model_count:
            best_model_count[best_model_name] += 1
        else:
            best_model_count[best_model_name] = 1
    print(f"############################## {exp_name} ##############################")
    print("Best model count according to BIC", best_model_count)

    ## sort by BIC
    df = df.sort_values(by=["BIC"])
    df["BIC"] = df["BIC"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    print("Grouped model and BIC")
    sorted_df = grouped_df.sort_values(by=["BIC"])
    print(sorted_df)
    return None


def classify_participants_mean_score(exp_name, pid_class, kind):
    # classify pariticpants into adaptive, maladaptive, mod. adaptive
    # if "single", then get classificaiton for a single condition
    # get score from mouselab mdp
    df = pd.read_csv(f"../data/human/{exp_name}/mouselab-mdp.csv")

    # sum score of all trials
    pid_score = df.groupby("pid")["score"].sum().to_frame()  # output is series(pid; score)

    # cluster according to score
    temp_list = pid_score.iloc[:, 0].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(temp_list)
    pid_score["label"] = kmeans.labels_

    cluster_centers = pd.Series(kmeans.cluster_centers_.flatten())
    cluster_centers = cluster_centers.sort_values()
    pid_score["label"] = pid_score["label"].replace(int(cluster_centers.index[0]), "maladaptive")
    pid_score["label"] = pid_score["label"].replace(int(cluster_centers.index[1]), "mod. adaptive")
    pid_score["label"] = pid_score["label"].replace(int(cluster_centers.index[2]), "adaptive")

    adaptive_pid = pid_score.index[pid_score['label'] == "adaptive"].tolist()
    maladaptive_pid = pid_score.index[pid_score['label'] == "maladaptive"].tolist()
    other_pid = pid_score.index[pid_score['label'] == "mod. adaptive"].tolist()

    if kind == "all":
        pid_class["adaptive"].append(adaptive_pid)
        pid_class["maladaptive"].append(maladaptive_pid)
        pid_class["mod. adaptive"].append(other_pid)
        return pid_class
    else:
        pid_dict = {}
        pid_dict["adaptive"] = adaptive_pid
        pid_dict["maladaptive"] = maladaptive_pid
        pid_dict["mod. adaptive"] = other_pid
        return pid_dict


def classify_participants_score_improvement(exp_name, pid_list, pid_class):
    # creates a list of pid whose score significantly improved during the trials
    df = pd.read_csv(f"../data/human/{exp_name}/mouselab-mdp.csv")

    df_pivot = pd.pivot_table(df, index=["trial_index"], columns=["pid"], values=["score"], fill_value=0)
    df_pivot.columns = pid_list

    for column in df_pivot:
        result = mk.original_test(df_pivot[column])
        if result.trend == "increasing":
            pid_class["adaptive"].append(column)
        else:
            pid_class["non-adaptive"].append(column)
    return pid_class


def find_adaptive_particpants(exp_name):
    pid_dict = pd.read_pickle(f"../../results/cm/inferred_strategies/{exp_name}_training/strategies.pkl")
    learning_pid = []
    for key, values in pid_dict.items():
        if len(set(values)) > 1:
            learning_pid.append(key)
    return learning_pid


if __name__ == "__main__":
    # model_list = [27, 31, 59, 63, 91, 95, 123, 127, 155, 159, 187, 191, 411,
    #               415, 443, 447, 475, 479, 507, 511, 539, 543, 571, 575, 603,
    #               607, 635, 639, 667, 671, 699, 703, 731, 735, 763, 767, 987,
    #               991, 1019, 1023, 1051, 1055, 1083, 1087, 1115, 1119, 1147,
    #               1151, 1179, 1183, 1211, 1215, 1243, 1247, 1275, 1279, 1307,
    #               1311, 1339, 1343, 1563, 1567, 1595, 1599, 1627, 1631, 1659,
    #               1663, 1691, 1695, 1723, 1727, 1755, 1759, 1819, 1823, 1851,
    #               1855, 1915, 1918, 1919, 1947, 1951, 2011, 2015, 5134]

    # model_list = [13, 15, 29, 31, 34, 35, 38, 39, 42, 43, 46, 47, 61, 63, 77, 79, 82, 83, 86, 87, 90, 91, 94, 95, 109, 111,
    #               125, 127, 130, 131, 134, 135, 138, 139, 142, 143, 157, 159, 173, 175, 178, 179, 182, 183, 186, 187, 190, 191,
    #               205, 207, 221, 223, 226, 227, 230, 231, 234, 235, 238, 239, 253, 255, 269, 271, 274, 275, 278, 279, 282, 283,
    #               286, 287, 301, 303, 317, 319, 322, 323, 326, 327, 330, 331, 334, 335, 349, 351, 365, 367, 370, 371, 374, 375,
    #               378, 379, 382, 383, 397, 399, 413, 415, 418, 419, 422, 423, 426, 427, 430, 431, 445, 447, 461, 463, 477, 479,
    #               482, 483, 486, 487, 490, 491, 494, 495, 498, 499, 502, 503, 1308]

    model_list = [13,
                  15,
                  29,
                  31,
                  34,
                  35,
                  38,
                  39,
                  42,
                  43,
                  46,
                  47,
                  109,
                  111,
                  125,
                  127,
                  130,
                  131,
                  134,
                  135,
                  138,
                  139,
                  142,
                  143,
                  157,
                  159,
                  173,
                  175,
                  178,
                  179,
                  182,
                  183,
                  186,
                  187,
                  190,
                  191,
                  253,
                  255,
                  269,
                  271,
                  274,
                  275,
                  278,
                  279,
                  282,
                  283,
                  286,
                  287,
                  301,
                  303,
                  317,
                  319,
                  322,
                  323,
                  326,
                  327,
                  330,
                  331,
                  334,
                  335,
                  397,
                  399,
                  413,
                  415,
                  418,
                  419,
                  422,
                  423,
                  426,
                  427,
                  430,
                  431,
                  445,
                  447,
                  477,
                  479,
                  482,
                  483,
                  490,
                  1743,
                  491,
                  494,
                  495,
                  502,
                  503,
                  1756]

    learning_participants_planning_amount = {
        "high_variance_high_cost": [32, 49, 57, 60, 94, 108, 109, 111, 129, 134, 139, 149, 161, 164, 191, 195],
        "high_variance_low_cost": [7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 145,
                                   146, 154, 158, 180, 189, 197],
        "low_variance_high_cost": [2, 13, 14, 24, 28, 36, 45, 61, 62, 69, 73, 79, 80, 86, 100, 107, 124, 128, 135, 138,
                                   160, 166, 171, 174, 183, 201, 206],
        "low_variance_low_cost": [9, 42, 52, 85, 110, 115, 143, 165, 172]}
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
    learning_participants = {"threecond": learning_participants_three_cond,
                             "planning": learning_participants_planning_amount}

    # learning_participants = { #here the planning amount exp are including maladpative participants, whose clicks changed
    #     "high_variance_high_cost": [32, 49, 57, 60, 94, 108, 109, 111, 129, 134, 139, 149, 161, 164, 191, 195, 30, 41, 46, 88, 116, 125, 156, 167, 173, 188, 204],
    #     "high_variance_low_cost": [7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 145,
    #                                146, 154, 158, 180, 189, 197, 29, 87, 131, 133, 175, 187],
    #     "low_variance_high_cost": [2, 13, 14, 24, 28, 36, 45, 61, 62, 69, 73, 79, 80, 86, 100, 107, 124, 128, 135, 138,
    #                                160, 166, 171, 174, 183, 201, 206],
    #     "low_variance_low_cost": [9, 42, 52, 85, 110, 115, 143, 165, 172, 5, 59, 67, 72, 127, 152, 155, 207],
    #     "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
    #              82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
    #              160, 165, 169, 173],
    #     "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
    #              99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
    #              170],
    #     "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
    #              100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
    #              168, 171]}

    exp_num_list_planning = ["high_variance_high_cost",
                             "high_variance_low_cost",
                             "low_variance_high_cost",
                             "low_variance_low_cost"]
    exp_num_list_threecond = ["v1.0",
                              "c2.1",
                              "c1.1"]

    # df_all = pd.DataFrame()
    # df_learning = pd.DataFrame()
    # pid_all = []
    for key, values in learning_participants.items():
        df_all = pd.DataFrame()
        df_learning = pd.DataFrame()
        pid_all = []
        if key == "threecond":
            exp_num_list = exp_num_list_threecond
        else:
            exp_num_list = exp_num_list_planning
        for exp_name in exp_num_list:
            # find_adaptive_particpants(exp_name)
            # load csv with intermediate results_400_second
            df = pd.read_csv(f"results_400_second_fit/{exp_name}_intermediate_results.csv")
            # drop first column
            del df[df.columns[0]]
            # filter df for models to be considered (no vicarious learning, no monte carlo, no subjective cost)
            df = df[df['model'].isin(model_list)]

            # get pid
            pid_list = get_all_pid_for_env(exp_name)

            ## calculate the metrics
            # loss(exp_name, df, pid_list)
            # aic(exp_name, df, pid_list)
            # bic(exp_name, df, pid_list)

            df_all = df_all.append(df)
            pid_all.append(pid_list)

            # create df with only adaptive participants
            df_learning = df_learning.append(df[df['pid'].isin(values[exp_name])])


        # analysis for all
        # flatten list
        # pid_all_flat = [item for sublist in pid_all for item in sublist]
        # loss("all", df_all, pid_all_flat)
        # bic_all = bic(key, df_all, pid_all_flat)
        # aic("all", df_all, pid_all_flat)

        ## BIC according to participant classification
        adaptive_list = [item for sublist in values.values() for item in sublist]
        bic_learners = bic(key, df_learning, adaptive_list)

        # aic("learning", df_learning, adaptive_list)


