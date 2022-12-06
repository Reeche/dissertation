import pandas as pd
from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env
from sklearn.cluster import KMeans
import numpy as np


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


def bic(exp_name, df, pid_list):
    ## Get best model for each participant and count which one occured most often based on BIC

    # filter by pid list
    df = df[df["pid"].isin(pid_list)]

    # create bic table for matlab
    df_pivot = pd.pivot_table(df, index=["pid"], columns=["model"], values=["BIC"], aggfunc=np.sum, fill_value=0)
    df_pivot.to_csv(f"results/matlab_all_exp_bic_{exp_name}.csv")

    best_model_count = {}
    for pid in pid_list:
        data_for_pid = df[df["pid"] == pid]
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


def classify_participants(exp_name, pid_class, kind):
    # classify pariticpants into adaptive, maladaptive, mod. adaptive
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



if __name__ == "__main__":
    models_to_be_considered = [26, 27, 30, 31, 58, 59, 62, 63, 90, 91, 94, 95, 122, 123, 126, 127, 154,
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

    exp_num_list = ["v1.0",
                    "c2.1",
                    "c1.1",
                    "high_variance_high_cost",
                    "high_variance_low_cost",
                    "low_variance_high_cost",
                    "low_variance_low_cost"]

    # exp_num_list = ["v1.0"]

    df_all = pd.DataFrame()
    pid_all = []
    pid_class = {"adaptive": [],
                 "maladaptive": [],
                 "mod. adaptive": []}
    for exp_name in exp_num_list:
        # load csv with intermediate results
        df = pd.read_csv(f"results/{exp_name}_intermediate_results.csv")
        # drop first column
        del df[df.columns[0]]
        # filter df for models to be considered (no vicarious learning, no monte carlo, no subjective cost)
        df = df[df['model'].isin(models_to_be_considered)]

        # get pid
        pid_list = get_all_pid_for_env(exp_name)

        loss(exp_name, df, pid_list)
        # aic(exp_name, df, pid_list)
        # bic(exp_name, df, pid_list)

        # df_all = df_all.append(df)
        # pid_all.append(pid_list[0])

        # classify by adaptiveness
        # pid_dict_single = classify_participants(exp_name, pid_class, "single")
        # pid_dict_all = classify_participants(exp_name, pid_class, "all")


    # analysis for all
    # bic("all", df_all, pid_all)
    # aic("all", df_all, pid_all)

    # BIC according to participant classification
    # adaptive_list = [item for sublist in pid_dict_all["adaptive"] for item in sublist]
    # bic("adaptive", df_all, adaptive_list)
    #
    # maladaptive_list = [item for sublist in pid_dict_all["maladaptive"] for item in sublist]
    # bic("maladaptive", df_all, maladaptive_list)
    #
    # mod_adaptive_list = [item for sublist in pid_dict_all["mod. adaptive"] for item in sublist]
    # bic("mod. adaptive", df_all, mod_adaptive_list)



