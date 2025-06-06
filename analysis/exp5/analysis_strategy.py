import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import chi2_contingency,  chisquare
from itertools import groupby

def create_one_pkl_file():
    strategy_scores_all = {}
    for strategy in range(1,90):
        strategy_scores = pd.read_pickle(f"../../results/strategy_scores/existence_with_click/{strategy}_strategy_scores.pkl")
        strategy_scores_all = {**strategy_scores_all, **strategy_scores}

    strategy_clicks_all = {}
    for strategy in range(1,90):
        strategy_clicks = pd.read_pickle(f"../../results/strategy_scores/existence_with_click/{strategy}_number_of_clicks.pkl")
        strategy_clicks_all = {**strategy_clicks_all, **strategy_clicks}
    return strategy_scores_all, strategy_clicks_all

def cluster(participant_strategies):
    """
    group according to adaptive, maladaptive and other strategies
    Args:
        participant_strategies_: df of participants and their strategies
    Returns:
    """
    # get the score data
    strategy_scores, strategy_clicks = create_one_pkl_file()

    ### limit the strategies to the ones with less than 7 clicks
    strategy_df = pd.DataFrame.from_dict([strategy_scores, strategy_clicks])
    strategy_df = strategy_df.T

    ### clustering of the strategy score:
    # create df with columns strategy, value, label
    strategy_df.columns=["score", "click"]

    # make strategy_df start from 1 as well
    strategy_df.index = np.arange(1, len(strategy_df) + 1)



    #strategy_df = strategy_df[strategy_df.index.isin(limited_strategy_space)]

    strategy_values_list = strategy_df["score"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(strategy_values_list)
    strategy_df["label"] = kmeans.labels_
    # strategy_df["strategy"] = strategy_scores.keys()

    ## relabel the cluster centers
    cluster_centers = pd.Series(kmeans.cluster_centers_.flatten())
    cluster_centers = cluster_centers.sort_values()
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[0]), "maladaptive_strategies")
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[1]), "other_strategies")
    strategy_df["label"] = strategy_df["label"].replace(int(cluster_centers.index[2]), "adaptive_strategies")

    adaptive_strategies = list(strategy_df[strategy_df['label'] == "adaptive_strategies"].index)
    maladaptive_strategies = list(strategy_df[strategy_df['label'] == "maladaptive_strategies"].index)
    other_strategies = list(strategy_df[strategy_df['label'] == "other_strategies"].index)

    # adaptive under click limitation (8) constraint strategies, i.e. change the labelling if click are more than 8
    # adaptive_under_constraint = list(strategy_df[(strategy_df['label'] == "adaptive_strategies") & (strategy_df['click'] < 8)].index)
    adaptive_under_constraint = list(strategy_df[strategy_df['label'] == "adaptive_strategies"].index)
    adaptive_without_constraint = [x for x in adaptive_strategies if x not in adaptive_under_constraint]

    # match adaptive, maladaptive and other to the participant
    # participant_strategies = participant_strategies - 1  # because when calculating strategy score, strategy start at 0 but here it starts at 1
    participant_strategies = participant_strategies.replace(adaptive_under_constraint, "adaptive_less_than_8_clicks")
    participant_strategies = participant_strategies.replace(adaptive_without_constraint, "adaptive_with_more_than_8_clicks")
    participant_strategies = participant_strategies.replace(maladaptive_strategies, "maladaptive_strategies")
    participant_strategies = participant_strategies.replace(other_strategies, "other_strategies")

    return participant_strategies

def frequency_of_strategy_change_test(control, exp):
    # adaptive strategies
    control_change_count = 0
    for index, rows in control.iterrows():
        if rows["Pretraining"] != "adaptive_less_than_8_clicks" and rows["Test"] == "adaptive_less_than_8_clicks":
            control_change_count += 1

    exp_change_count = 0
    for index, rows in exp.iterrows():
        if rows["Pretraining"] != "adaptive_less_than_8_clicks" and rows["Test"] == "adaptive_less_than_8_clicks":
            exp_change_count += 1
    print(f"In experimental condition {exp_change_count} pid changed to an adaptive strategy.")
    print(f"In control condition {control_change_count} pid changed to an adaptive strategy.")
    print(f"Chi^2 test goodness of fit of strategy change: ", chisquare([control_change_count, exp_change_count]))

    # maladaptive strategies
    control_maladaptive_change_count = 0
    for index, rows in control.iterrows():
        if rows["Pretraining"] == "maladaptive_strategies" and rows["Test"] != "maladaptive_strategies":
            control_maladaptive_change_count += 1

    exp_maladaptive_change_count = 0
    for index, rows in exp.iterrows():
        if rows["Pretraining"] == "maladaptive_strategies" and rows["Test"] != "maladaptive_strategies":
            exp_maladaptive_change_count += 1
    print(f"In experimental condition {exp_maladaptive_change_count} pid changed away from a maladaptive strategy.")
    print(f"In control condition {control_maladaptive_change_count} pid changed away from a maladaptive strategy.")
    print(f"Chi^2 test goodness of fit of strategy change: ", chisquare([control_maladaptive_change_count, exp_maladaptive_change_count]))

    return None



experiment = "with_click"

### comparing exp pretraining and test, expect to see difference through training
pretraining_exp = pd.read_pickle(f"../../results/cm/inferred_strategies/existence_{experiment}_exp_pretraining/strategies.pkl")
test_exp = pd.read_pickle(f"../../results/cm/inferred_strategies/existence_{experiment}_exp_test/strategies.pkl")

pretraining_exp = pd.DataFrame.from_dict(pretraining_exp)
test_exp = pd.DataFrame.from_dict(test_exp)

# create dataframe
df_exp = pretraining_exp._append(test_exp)

labelled_strategies_exp = cluster(df_exp)
labelled_strategies_exp = labelled_strategies_exp.T
labelled_strategies_exp.columns = ['Pretraining', 'Test']


# count and calculate proportion of adaptive, maladaptive and other strategies
print("EXPERIMENTAL")
print(labelled_strategies_exp["Pretraining"].value_counts().sort_index())
print(labelled_strategies_exp["Test"].value_counts().sort_index())
obs_1 = np.array([labelled_strategies_exp["Pretraining"].value_counts().sort_index(), labelled_strategies_exp["Test"].value_counts().sort_index()])
chi2, p, dof, expctd = chi2_contingency(obs_1)
print(f"Chi square independence test between EXPERIMENTAL pretraining and test: chi2={chi2}, p={p}")


### comparing control pretraining and test, expect to see NO difference because NO training
pretraining_control = pd.read_pickle(f"../../results/cm/inferred_strategies/existence_{experiment}_control_pretraining/strategies.pkl")
test_control = pd.read_pickle(f"../../results/cm/inferred_strategies/existence_{experiment}_control_test/strategies.pkl")

pretraining_control = pd.DataFrame.from_dict(pretraining_control)
test_control = pd.DataFrame.from_dict(test_control)

df_control = pretraining_control._append(test_control)

labelled_strategies_control = cluster(df_control)
labelled_strategies_control = labelled_strategies_control.T
labelled_strategies_control.columns = ['Pretraining', 'Test']

# count and calculate proportion of adaptive, maladaptive and other strategies
print("CONTROL")
print(labelled_strategies_control["Pretraining"].value_counts().sort_index())
print(labelled_strategies_control["Test"].value_counts().sort_index())
a = labelled_strategies_control["Pretraining"].value_counts().sort_index()
b = labelled_strategies_control["Test"].value_counts().sort_index()
if len(a) != len(b):
    a["adaptive_more_than_8_clicks"] = 0
    print("check which strategy is missing and add accordingly")


# chi^2 test between controls
obs_2 = np.array([a, b])
chi2, p, dof, expctd = chi2_contingency(obs_2)
print(f"Chi square independence test between CONTROL pretraining and test: chi2={chi2}, p={p}")

# chi^2 test between exps
obs_5 = np.array([labelled_strategies_exp["Pretraining"].value_counts().sort_index(), labelled_strategies_exp["Test"].value_counts().sort_index()])
chi2, p, dof, expctd = chi2_contingency(obs_5)
print(f"Chi square independence test between EXP pretraining and test: chi2={chi2}, p={p}")

# chi^2 test between control vs exp
obs_3 = np.array([labelled_strategies_exp["Pretraining"].value_counts().sort_index(), a])
chi2, p, dof, expctd = chi2_contingency(obs_3)
print(f"Chi square independence test between CONTROL vs EXP pretraining: chi2={chi2}, p={p}")

# obs_4 = np.array([labelled_strategies_exp["Test"].value_counts().sort_index(), labelled_strategies_control["Test"].value_counts().sort_index()])
obs_4 = np.array([labelled_strategies_exp["Test"].value_counts().sort_index(), b])
chi2, p, dof, expctd = chi2_contingency(obs_4)
print(f"Chi square independence test between CONTROL vs EXP test: chi2={chi2}, p={p}")

# chi^2 test of goodness of fit between observed and expected (here the average)
frequency_of_strategy_change_test(labelled_strategies_control, labelled_strategies_exp)


# count the number of strategies used, strategies start from 1
unique, counts = np.unique(df_exp.iloc[0].values, return_counts=True)
print("Exp pretraining", dict(zip(unique, counts)))
unique, counts = np.unique(df_exp.iloc[1].values, return_counts=True)
print("Exp posttraining", dict(zip(unique, counts)))
unique, counts = np.unique(df_control.iloc[0].values, return_counts=True)
print("Control pretraining", dict(zip(unique, counts)))
unique, counts = np.unique(df_control.iloc[1].values, return_counts=True)
print("Control posttraining", dict(zip(unique, counts)))


# strategy sequence analysis

falks_clustering = {1: [32, 33, 34, 36],
                    2: [55, 56, 59, 51],
                    3: [60, 64, 65, 66, 69, 75, 48, 27, 26, 16, 19, 21, 24, 23, 6, 15],
                    4: [49, 50, 52, 18, 23],
                    5: [53, 54, 57, 61, 58, 70, 71, 35, 30, 72],
                    6: [73, 31, 28, 29, 1, 2, 3, 4, 5, 63],
                    7: [62, 67, 17, 23, 7, 8, 9, 10, 11, 12, 13, 14, 25, 78, 79],
                    8: [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 20, 22, 68, 74, 76, 77]}
                    # 9: [68, 74, 76, 77]}
cluster_name_mapping = {1: "P - Best first search",
                        2: "P - Near-sighted planning",
                        3: "P - Far-sighted planning",
                        4: "NP - Frugal planning",
                        5: "NP - Near-sighted planning",
                        6: "NP - Exhaustive planning",
                        7: "NP - Far-sighted planning",
                        # 8: "Non-Prioritizing - Local search",
                        8: "Misc. planning"}

# strategy_maping = pd.read_pickle("../../mcl_toolbox/data/si_strategy_map.pkl")
strategy_maping = pd.read_pickle("../../mcl_toolbox/data/si_strategy_names.pkl")
# replace the straegy number from Falk's clustering with the ones implemented
for key, value in falks_clustering.items():
    falks_clustering[key] = [strategy_maping.get(item, item) for item in value]

# make falks clustering in a shape of {"strategy number": "strategy name"}
cluster_mapping = {}
for k, v in falks_clustering.items():
    for x in v:
        cluster_mapping.setdefault(x, []).append(k)

# remove the list in the values
for key, value in cluster_mapping.items():
    cluster_mapping[key] = value[0]


df_exp_mapped_to_cluster = df_exp.replace(cluster_mapping)
df_exp_mapped_to_cluster = df_exp_mapped_to_cluster.replace(cluster_name_mapping)
# df_exp_mapped_to_cluster = df_exp_mapped_to_cluster.replace(clustering_the_cluster)

df_control_mapped_to_cluster = df_control.replace(cluster_mapping)
df_control_mapped_to_cluster = df_control_mapped_to_cluster.replace(cluster_name_mapping)
# df_control_mapped_to_cluster = df_control_mapped_to_cluster.replace(clustering_the_cluster)

def analyse_strategy_sequence(data, condition):
    # find out how often a trajectory has been used
    # this creates a list of unique values, i.e. a -> b -> a will result in a -> b
    participants_dict_unique_values = {}
    for columns in data:
        # only tuples are hashable for flipping
        participants_dict_unique_values[columns] = tuple(data[columns].unique())

    # this removes consecutive same values, i.e. a -> b -> a will result in a -> b -> a
    participants_dict = {}
    for columns in data:
        participants_dict[columns] = tuple([i[0] for i in groupby(data[columns])])

    # flip the dict, so the value now contains which pids used the trajectory
    flipped = {}
    for key, value in participants_dict.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)

    number_of_participants = data.shape[1]

    # replace pids for a certain trajectory with their count and divide by total number of participants count
    for key, value in flipped.items():
        flipped[key] = len(value) / number_of_participants

    # sort flipped to get the trajectories with highest probabilities
    sorted_results = {k: v for k, v in sorted(flipped.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_results)

    # count frequency of people who switched to far-sighted
    far_sighted_counter = 0
    for key, value in participants_dict.items():
        if len(value) > 1 and "P - Far-sighted planning" in value:
            far_sighted_counter += 1
        if len(value) > 1 and "NP - Far-sighted planning" in value:
            far_sighted_counter += 1

    res = pd.DataFrame(columns=["usage", "sequence"])
    res["usage"] = sorted_results.values()
    res["sequence"] = sorted_results.keys()
    # res.to_csv(f"results/{experiment}_{condition}_sequences.csv")
    return far_sighted_counter

#pretraining: increasing variance, posttraining: increasing variance
far_sighted_count_exp = analyse_strategy_sequence(df_exp_mapped_to_cluster, "exp")
far_sighted_count_control = analyse_strategy_sequence(df_control_mapped_to_cluster, "control")

print(f"{far_sighted_count_exp} participants switched to a far-sighted strategy in experimental condition")
print(f"{far_sighted_count_control} participants switched to a far-sighted strategy in control condition")
print(f"Chi^2 test goodness of fit of strategy change: ", chisquare([far_sighted_count_exp, far_sighted_count_control]))