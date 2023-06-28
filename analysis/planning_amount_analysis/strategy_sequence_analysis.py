import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# high_variance_low_cost
# high_variance_high_cost
# low_variance_low_cost
# low_variance_high_cost

experiment = "low_variance_low_cost"
# add cluster names
cluster_name_mapping = {1: "Goal-setting with exhaustive backward planning",
                        2: "Forward planning strategies similar to Breadth First Search",
                        3: "Middle-out planning",
                        4: "Forward planning strategies similar to Best First Search",
                        5: "Local search",
                        6: "Maximizing Goal-setting with exhaustive backward planning",
                        7: "Frugal planning",
                        8: "Myopic planning",
                        9: "Maximizing goal-setting with limited backward planning",
                        10: "Frugal goal-setting strategies",
                        11: "Strategy that explores immediate outcomes on the paths to the best final outcomes",
                        12: "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing",
                        13: "Miscellaneous  strategies"}

training = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")

training_cluster_df = pd.DataFrame.from_dict(training)

# map strategy to cluster
cluster_mapping = pd.read_pickle(f"../../mcl_toolbox/data/kl_cluster_map.pkl")
training_cluster_df = training_cluster_df.replace(cluster_mapping)
# training_cluster_df = training_cluster_df.replace(cluster_name_mapping)


def cluster_the_cluster(training_cluster_df):
    """
    To find out how to cluster the cluster according to the number of clicks

    Args:
        training_cluster_df:

    Returns:

    """
    # import number of click information
    strategy_clicks = pd.read_pickle(f"results/cm/strategy_scores/{experiment}_numberclicks.pkl")
    strategy_clicks_df = pd.DataFrame.from_dict(strategy_clicks, orient="index")
    strategy_clicks_df = strategy_clicks_df.reset_index()  # adds "index" column
    strategy_clicks_df.columns = ["index", "clicks"]
    strategy_clicks_df["index"] = np.arange(1, len(strategy_clicks_df) + 1)
    strategy_clicks_df["index"] = strategy_clicks_df["index"].replace(cluster_mapping)

    # click - 1
    strategy_clicks_df["clicks"] -= 1

    # use k-means to group the clicks into n clusters
    n = 13
    click_values = strategy_clicks_df["clicks"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(click_values)
    strategy_clicks_df["label"] = kmeans.labels_

    # replace the cluster number with actual names
    strategy_clicks_df["index"] = strategy_clicks_df["index"].replace(cluster_name_mapping)

    # remove the ones that did not get a label as those are the 10 strategies not frequently used, i.e. not in CM
    strategy_clicks_df = strategy_clicks_df[strategy_clicks_df["index"].apply(lambda x: isinstance(x, str))]

    # count how often a certain cluster has been assigned with a k-means cluster
    temp_dict = {}
    for key, value in cluster_name_mapping.items():
        temp = strategy_clicks_df[strategy_clicks_df["index"] == value]
        # count the most frequent label
        temp_label = max(set(list(temp["label"].values)), key=list(temp["label"].values).count)
        temp_dict[value] = temp_label
    print(strategy_clicks_df.groupby(["index"])["clicks"].mean())
    # look at temp_dict to manually get the cluster #todo make it automatic
    return strategy_clicks_df


# cluster_the_cluster(training_cluster_df)

# regardless of experiment condition, the clustering and the average number of clicks for the clusters remain the same
# clustering_the_cluster = {"Goal-setting with exhaustive backward planning": "12 clicks",
#                           "Forward planning strategies similar to Breadth First Search": "12 clicks",
#                           "Middle-out planning": "12 clicks",
#                           "Forward planning strategies similar to Best First Search": "12 clicks",
#                           "Local search": "12 clicks",
#                           "Maximizing Goal-setting with exhaustive backward planning": "12 clicks",
#                           "Frugal planning": "0 clicks",
#                           "Myopic planning": "5 clicks",
#                           "Maximizing goal-setting with limited backward planning": "5 clicks",
#                           "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing": "5 clicks",
#                           "Frugal goal-setting strategies": "3 clicks",
#                           "Strategy that explores immediate outcomes on the paths to the best final outcomes": "8 clicks"}

clustering_the_cluster = {"Goal-setting with exhaustive backward planning": "Goal-setting",
                          "Forward planning strategies similar to Breadth First Search": "Forward planning",
                          "Middle-out planning": "Middle-out planning",
                          "Forward planning strategies similar to Best First Search": "Forward planning",
                          "Local search": "Local search",
                          "Maximizing Goal-setting with exhaustive backward planning": "Goal-setting",
                          "Frugal planning": "Frugal planning",
                          "Myopic planning": "Little planning",
                          "Maximizing goal-setting with limited backward planning": "Goal-setting",
                          "Frugal goal-setting strategies": "Goal-setting",
                          "Strategy that explores immediate outcomes on the paths to the best final outcomes": "Final and then immediate outcome",
                          "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing": "Final and then immediate outcome",
                          "Miscellaneous strategies": "Miscellaneous strategies"}

# cluster the cluster
# training_cluster_df = training_cluster_df.replace(clustering_the_cluster)

# find out how often a trajectory has been used
def trajectory_frequency(training_cluster_df):
    participants_dict = {}
    for columns in training_cluster_df:
        # only tuples are hashable for flipping
        participants_dict[columns] = tuple(training_cluster_df[columns].unique())

    # flip the dict, so the value now contains which pids used the trajectory
    flipped = {}
    for key, value in participants_dict.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)

    number_of_participants = len(training)

    # replace pids for a certain trajectory with their count and divide by total number of participants count
    for key, value in flipped.items():
        flipped[key] = len(value) / number_of_participants

    # sort flipped to get the trajectories with highest probabilities
    sorted_results = {k: v for k, v in sorted(flipped.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_results)

"""
those clicks stay approx the same for all 4 conditions
Average clicking numbers: for high_variance_high_cost
Forward planning strategies similar to Best First Search                                               9.618811 
Forward planning strategies similar to Breadth First Search                                           11.282074
Frugal goal-setting strategies                                                                         3.330498
Frugal planning                                                                                        1.379657
Goal-setting with exhaustive backward planning                                                        12.000000
Local search                                                                                           9.449656
Maximizing Goal-setting with exhaustive backward planning                                             12.000000
Maximizing goal-setting with limited backward planning                                                 4.806002
Middle-out planning                                                                                   12.000000
Miscellaneous  strategies                                                                              8.075122
Myopic planning                                                                                        4.661056
Strategy that explores immediate outcomes on the paths to the best final outcomes                      7.542880
Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing     4.549273
"""

## how many people did not change strategy type but changed number of clicks
print(2)