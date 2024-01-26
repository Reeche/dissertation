import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import statsmodels.api as sm
from vars import clicking_pid
import pymannkendall as mk


def plot_strategy_proportions(data, mapping: dict):
    df = data.replace(mapping)

    frequencies = pd.DataFrame(columns=["Adaptive", "Mod. adaptive", "Maladaptive"])
    temp_freq = {}
    for columns in df:
        temp_freq["Adaptive"] = Counter(df[columns])["adaptive"] / len(data[0])
        temp_freq["Mod. adaptive"] = Counter(df[columns])["mod"] / len(data[0])
        temp_freq["Maladaptive"] = Counter(df[columns])["mal"] / len(data[0])
        frequencies = frequencies._append(temp_freq, ignore_index=True)

    trend_test(frequencies)
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

def logistic_regression(participants_df, mapping_dict):

    # replace the column strategy with the mapping selected from the column condition
    participants_df["strategy"] = participants_df.apply(lambda x: mapping_dict[x["condition"]][x["strategy"]], axis=1)

    # replace strategy with 1 for adaptive and 0 for anything else
    participants_df["strategy"] = [1 if x == "adaptive" else 0 for x in participants_df["strategy"]]

    # logistic regression
    # model = sm.GLM.from_formula("strategy ~ C(condition, Treatment('c1.1'))*trial", data=participants_df,
    #                             family=sm.families.Binomial()).fit()
    # print(model.summary())

    ### logisit regression for each condition
    for condition in pd.unique(participants_df["condition"]):
        condition_df = participants_df[participants_df["condition"] == condition]
        model = sm.GLM.from_formula("strategy ~ trial", data=condition_df,
                                    family=sm.families.Binomial()).fit()
        print(condition)
        print(model.summary())

    # from sklearn.linear_model import LogisticRegression
    # logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
    # # x_train is trial and condition, convert condition to dummy variables
    # X_train = pd.get_dummies(participants_df[["trial", "condition"]], columns=["condition"])
    #
    # # y_train is strategy
    # y_train = participants_df["strategy"]
    # logreg.fit(X_train, y_train)
    # print(logreg.coef_)
    return None

def trend_test(data):
    result = mk.original_test(data["Adaptive"])
    print(result)
    return None

if __name__ == "__main__":
    # experiment = "v1.0"
    experiments = ["v1.0", "c2.1", "c1.1"]
    all_pid = pd.DataFrame()
    mapping_dict = {}
    for experiment in experiments:
        strategy_scores = pd.read_pickle(f"../../results/strategy_scores/{experiment}_strategy_scores.pkl")
        strategy_scores = pd.DataFrame.from_dict(strategy_scores, orient='index')
        strategy_scores.index += 1  # increment by one because participants start at 1
        participants = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")

        # filter for clicking participants
        participants = {key: value for key, value in participants.items() if key in clicking_pid[experiment]}

        participants_df = pd.DataFrame.from_dict(participants, orient='index')

        ## get only used strategies
        unique_used_strategies = pd.unique(participants_df.values.flatten())

        ## filter strategy score by used strategies
        used_strategy_score = strategy_scores.loc[unique_used_strategies] #important to use loc here

        strategy_labels = clustering(used_strategy_score)

        ## create mapping
        mapping = used_strategy_score.set_index(strategy_labels.index)['label']

        # adaptive_proportion_higher_than_chance(strategy_labels, participants_df)
        plot_strategy_proportions(participants_df, mapping)

        ### reshape df for logisitic regression
        # participants_df = participants_df.transpose()
        # participants_df = pd.melt(participants_df)
        # participants_df.columns = ["pid", "strategy"]
        # participants_df["trial"] = participants_df.groupby("pid").cumcount() + 1
        # participants_df["condition"] = experiment
        #
        # all_pid = all_pid._append(participants_df)
        # mapping_dict[experiment] = mapping
    #
    # logistic_regression(all_pid, mapping_dict)

