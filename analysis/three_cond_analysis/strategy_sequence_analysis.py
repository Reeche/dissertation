import pandas as pd
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk
from scipy import stats
from sklearn.cluster import KMeans
from mcl_toolbox.utils.learning_utils import get_participant_scores

experiment = "v1.0"
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


def magnitude_of_change_based_on_cluster(strategy_df, type):
    # How often are changes within one cluster and outside one cluster?
    # count how many unique values within 35 trials
    results = []
    for columns in strategy_df:
        results.append(strategy_df[columns].nunique())

    # count how many changes in total
    print(experiment, sum(results))

    # plt.hist(results, bins=13, range=(0, 13))
    # plt.xlabel("Number of changes")
    # plt.ylabel("Count of changes")
    # plt.savefig(f"plots/{experiment}_magnitude_of_change_{type}.png")
    # plt.show()
    # plt.close()


def magnitude_of_change_based_on_jeffrey(strategy_df, between_cluster_average, within_cluster_average, ):
    jeffrey_table = pd.read_pickle(f"../../mcl_toolbox/data/jeffreys_divergences.pkl")
    jeffrey_table = pd.DataFrame(jeffrey_table)

    # get average of divergence for all strategies
    # average_divergence = jeffrey_table.median().median() / 2
    average_divergence = within_cluster_average

    ##create df with jeff divergence
    df = strategy_df.copy()
    df = df - 1  # all strategy index values -1 because jeffrey table start at 0

    for column in df:
        for i in range(len(df[column]) - 2 + 1):
            strategy_a, strategy_b = df[column][i: i + 2]
            # look up jeff divergence from the table
            df[column][i] = jeffrey_table.loc[strategy_a][strategy_b]

    # # drop the last row because there is nothing to compare against
    df = df.drop(index=df.index[-1], axis=0)

    # all values to a long list
    all_values = df.values.tolist()
    all_values = [item for sublist in all_values for item in sublist]

    # drop all 0
    non_zero_values = [i for i in all_values if i != 0]
    print("Larger than within-cluster average: ", sum(i > average_divergence for i in non_zero_values))
    print("Smaller than within-cluster average: ", sum(i < average_divergence for i in non_zero_values))

    ##plot histogram with mean as vertical line
    # plt.hist(all_values, bins='auto', range=[0, 1000])
    # plt.axvline(x=average_divergence, color='b', label='Within cluster average')
    # plt.xlabel("Jeffrey divergence")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.savefig(f"plots/{experiment}_jeffrey_hist_absolute.png")
    # # plt.show()
    # plt.close()

    # count how many before the line and how many after the line
    # print("Smaller than average: ", df[df < average_divergence].count().sum())
    # print("Larger than average: ", df[df > average_divergence].count().sum())

    # plot jeffey development over time
    # trimmed_mean = stats.trim_mean(df, 0.1, axis=1)
    # plt.plot(trimmed_mean)
    # print(average_divergence)
    # plt.axhline(y=average_divergence, color='r', linestyle='-', label='Average')
    # plt.ylim(0, 2400)
    # plt.xlabel("Trials")
    # plt.ylabel("Jeffrey divergence")
    # ci = 1.96 * np.std(df.T) / np.sqrt(len(df.T))
    # plt.fill_between(range(len(trimmed_mean)), trimmed_mean - ci, trimmed_mean + ci, color="b", alpha=.1)
    # plt.legend()
    # plt.savefig(f"plots/{experiment}_jeffrey_development_trimmed_10.png")
    # # plt.show()
    # plt.close()

    # jeffrey trend
    # result = mk.original_test(non_zero_values)
    # print(f"Mann Kendall test for clicks for {experiment}: clicks are {result}")




def get_absolute_jeffrey_values(cluster_mapping):
    # find the average jeffrey difference within one cluster and jeffrey difference between clusters
    jeffrey_table = pd.read_pickle(f"../../mcl_toolbox/data/jeffreys_divergences.pkl")
    jeffrey_table = pd.DataFrame(jeffrey_table)

    # replace the jeffrey table column names (strategy) with corresponding cluster
    # note jeffrey table start at 0, cluster_mapping start at 1
    jeffrey_table.columns = range(1, 90) #columns
    jeffrey_table.index += 1 #rows


    # find the strategies that belong within one cluster and create a sub-df
    res = defaultdict(list)
    for key, val in sorted(cluster_mapping.items()):
        res[val].append(key)

    withtin_cluster_jeffrey = {}
    for key, val in res.items():
        # filter columns
        jeffrey_sub_table_within = jeffrey_table[val]
        # filter row
        jeffrey_sub_sub_table_within = jeffrey_sub_table_within[jeffrey_sub_table_within.index.isin(val)]
        ## take the median after removing the 0; median can work with the mirror structure
        stacked_within_values = jeffrey_sub_sub_table_within.values.tolist()
        if len(stacked_within_values) == 1: #only 0 left in the table
            withtin_cluster_jeffrey[key] = 0
        else:
            withtin_cluster_jeffrey[key] = statistics.median(stacked_within_values[stacked_within_values != 0])


    # calculate the jeffrey between the strategies from different clusters
    between_cluster_jeffrey = {}
    for key1, val1 in res.items():
        jeffrey_sub_table_between = jeffrey_table[val1]
        for key2, val2 in res.items():
            if key2 != key1: #create df with one set of strategy on the x and the other strategies on the y
                # no repeating strategies, therefore does not need to filter for 0 and no mirror structure
                jeffrey_sub_sub_table_between = jeffrey_sub_table_between[jeffrey_sub_table_between.index.isin(val2)]
                key_value = '-'.join(str(x) for x in [key1, key2])
                stacked_between_values = jeffrey_sub_sub_table_between.values.tolist()
                flat_list = [item for sublist in stacked_between_values for item in sublist]
                between_cluster_jeffrey[key_value] = statistics.median(flat_list)

    print("between cluster average", sum(between_cluster_jeffrey.values()) / len(between_cluster_jeffrey))
    print("within cluster average", sum(withtin_cluster_jeffrey.values()) / len(withtin_cluster_jeffrey))
    return sum(between_cluster_jeffrey.values()) / len(between_cluster_jeffrey), sum(withtin_cluster_jeffrey.values()) / len(withtin_cluster_jeffrey)

between_cluster_average, within_cluster_average = get_absolute_jeffrey_values(cluster_mapping)
magnitude_of_change_based_on_jeffrey(strategy_unclustered, between_cluster_average, within_cluster_average)


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
