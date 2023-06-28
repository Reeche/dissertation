import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import chisquare


def plot_strategy_proportions(data, mapping: dict):
    df = data.replace(mapping)

    frequencies = pd.DataFrame(columns=["Adaptive", "Mod. adaptive", "Maladaptive"])
    temp_freq = {}
    for columns in df:
        temp_freq["Adaptive"] = Counter(df[columns])["adaptive"] / len(data[0])
        temp_freq["Mod. adaptive"] = Counter(df[columns])["mod"] / len(data[0])
        temp_freq["Maladaptive"] = Counter(df[columns])["mal"] / len(data[0])
        frequencies = frequencies.append(temp_freq, ignore_index=True)

    plt.plot(frequencies)
    # plt.legend()
    plt.show()
    plt.close()
    return None


def compare_last_trial_proportions(participants: dict):
    ## get last row of strategies
    last_strategy = []
    for key, value in participants.items():
        if isinstance(value, list) and len(value) > 0:
            last_strategy.append(value[-1])
    # Use map() function to replace each item from the list
    mapped_list = list(map(mapping_dict.get, last_strategy))
    counter = Counter(mapped_list)
    print("Participants")
    for key in counter.keys():
        print(key, counter[key] / len(mapped_list))


def clustering(strategy_scores):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(strategy_scores.values)
    strategy_scores["label"] = kmeans.labels_

    ## relabel the cluster centers
    cluster_centers = pd.Series(kmeans.cluster_centers_.flatten())
    cluster_centers = cluster_centers.sort_values()
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[0]), "mal")
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[1]), "mod")
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[2]), "adaptive")
    return strategy_scores


def frequency(kmeans_results):
    counter = Counter(kmeans_results.labels_)
    print("Simulation")
    for key in counter.keys():
        print(key, counter[key] / len(kmeans_results.labels_))

def adaptive_proportion_higher_than_chance(strategy_labels, participants_df):
    ## count how many
    cm_counts = (strategy_labels["label"].value_counts() / sum(strategy_labels["label"].value_counts())) * 100
    participants_counter = participants_df.iloc[:,-1:].replace(mapping_dict)
    participant_counts = (participants_counter.value_counts() / sum(participants_counter.value_counts())) * 100
    non_adaptive_cm = cm_counts["mal"] + cm_counts["mod"]
    if "mal" in participant_counts.index and "mod" in participant_counts.index:
        non_adaptive_pid = participant_counts["mal"] + participant_counts["mod"]
    elif "mal" in participant_counts.index and "mod" not in participant_counts.index:
        non_adaptive_pid = participant_counts["mal"]
    elif "mod" in participant_counts.index and "mal" not in participant_counts.index:
        non_adaptive_pid = participant_counts["mod"]
    print("pid", participant_counts["adaptive"], non_adaptive_pid)
    print("CM", cm_counts["adaptive"], non_adaptive_cm)
    res = chisquare([participant_counts["adaptive"], non_adaptive_pid], f_exp=[cm_counts["adaptive"], non_adaptive_cm])
    print(res)

if __name__ == "__main__":
    # experiment = "v1.0"
    experiments = ["v1.0", "c2.1", "c1.1"]
    for experiment in experiments:
        strategy_scores = pd.read_pickle(f"../../results/strategy_scores/{experiment}_strategy_scores.pkl")
        strategy_scores = pd.DataFrame.from_dict(strategy_scores, orient='index')
        strategy_scores.index += 1  # increment by one because participants start at 1
        participants = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")

        participants_df = pd.DataFrame.from_dict(participants, orient='index')

        ## get only used strategies
        unique_used_strategies = pd.unique(participants_df.values.flatten())

        ## filter strategy score by used strategies
        used_strategy_score = (strategy_scores.loc[unique_used_strategies])

        strategy_labels = clustering(used_strategy_score)
        ## create mapping
        mapping_dict = used_strategy_score.set_index(strategy_labels.index)['label']

        adaptive_proportion_higher_than_chance(strategy_labels, participants_df)

        # plot_strategy_proportions(participants_df, mapping_dict)
