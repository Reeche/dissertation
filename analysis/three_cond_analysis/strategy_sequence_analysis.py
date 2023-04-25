import pandas as pd
import matplotlib.pyplot as plt

experiment = "c1.1"
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
strategy_unclustered = pd.DataFrame.from_dict(training)
strategy_cluster_df = pd.DataFrame.from_dict(training)

# map strategy to cluster
cluster_mapping = pd.read_pickle(f"../../mcl_toolbox/data/kl_cluster_map.pkl")
strategy_cluster_df = strategy_cluster_df.replace(cluster_mapping)
strategy_cluster_df = strategy_cluster_df.replace(cluster_name_mapping)


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
strategy_cluster_cluster_df = strategy_cluster_df.replace(clustering_the_cluster)

# magnitude of changes: for this we need the unclustered strategies.
# How often are changes within one cluster and outside one cluster?
def magnitude_of_change(strategy_cluster_df):
    # count how many uniquue values within 35 trials
    results = []
    for columns in strategy_cluster_df:
        results.append(strategy_cluster_df[columns].nunique())

    plt.hist(results, bins='auto')
    plt.xlabel("Number of strategy cluster changes")
    plt.ylabel("Count of strategy cluster changes")
    plt.savefig(f"{experiment}_magnitude_of_change.png")
    # plt.show()
    plt.close()

magnitude_of_change(strategy_cluster_df)

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

