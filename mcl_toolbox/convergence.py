import operator
import sys
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

import numpy as np

from mcl_toolbox.utils import learning_utils, distributions, analysis_utils
sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

"""
Idea: make this one a general analysis file, which contains several analysis 

1. Percentage of participants that changed their strategy and strategy cluster
2. Average scores 
3. Average number of clicks

Format: python3 convergence.py <reward_structure> <block> 
Example: python3 convergence.py increasing_variance training 
"""


def remove_duplicates(cluster_list):
    previous_cluster = cluster_list[0]
    non_duplicate_list = [previous_cluster]
    duplicate_freqs = [1]
    for i in range(1,len(cluster_list)):
        if cluster_list[i] != previous_cluster:
            non_duplicate_list.append(cluster_list[i])
            previous_cluster = cluster_list[i]
            duplicate_freqs.append(1)
        else:
            duplicate_freqs[-1]+= 1
    res = (tuple(non_duplicate_list), tuple(duplicate_freqs))
    return res

def get_sorted_trajectories(cluster_map, strategies):
    """
    Assign frequency to cluster and strategies
    Args:
        strategies: load from strategies.pkl

    Returns: cluster and strategy list in list, where first list describes the cluster/strategy and second list describes
    the frequency. Example: [((39, 37, 61, 39, 37, 39, 37, 39, 33, 49, 39), (1, 2, 1, 3, 4, 1, 4, 5, 1, 1, 12)), 1]
    Reads: strategy 39 was used for 1 trials, then strategy 37 was used for 2 trials

    """
    cluster_trajectory_frequency = defaultdict(int)
    strategy_trajectory_frequency = defaultdict(int)
    for pid, strategy_sequence in strategies.items():
        cluster_strategy_sequence = [cluster_map[strategy] for strategy in strategy_sequence]
        cluster_trajectory = remove_duplicates(cluster_strategy_sequence)
        strategy_trajectory = remove_duplicates(strategy_sequence)
        cluster_trajectory_frequency[cluster_trajectory] += 1
        strategy_trajectory_frequency[strategy_trajectory] += 1
    sorted_cluster_trajectory = [list(s) for s in sorted(cluster_trajectory_frequency.items(), key=operator.itemgetter(1),reverse = True)]
    sorted_strategy_trajectory = [list(s) for s in sorted(strategy_trajectory_frequency.items(), key=operator.itemgetter(1),reverse = True)]
    return sorted_cluster_trajectory, sorted_strategy_trajectory

def analyze_trajectory(trajectory, print_trajectories=False):
    final_repetition_count = []
    for tr in trajectory:
        if len(tr[0]) > 1:
            if print_trajectories:
                print("Trajectory:", tr[0][0])
                print("Repetition Frequency:", tr[0][1])
                print("Freq:", tr[1], "\n")
            temp = list(tr[0][1])
            temp.pop()
            number_of_trials_before_last_trial = np.sum(temp)
            #final_repetition_count.append(tr[0][1][-1])
            final_repetition_count.append(number_of_trials_before_last_trial)
            #print("The last item in Repetition Frequency", tr[0][1][-1])

    average_trials_repetition = np.mean(final_repetition_count)
    median_trials_repetition = np.median(final_repetition_count)
    print("Median final strategy usage: ", median_trials_repetition)
    print("Mean final strategy usage:", average_trials_repetition)

def plot_difference_between_trials(cluster_map, strategies: defaultdict, number_participants, cluster=False):
    """
    It creates a plot which shows the percentage of participants who changed their strategy across trial
    Args:
        strategies: A list of strategies for all participants (index is pid) across trials.
        number_participants: fixed number of participants

    Returns: two plots, one that plots percentage of participants that changed their strategy and strategy cluster

    """
    change_list_of_dicts = []
    for key, value in strategies.items():
        if cluster:
            # mapping strategy to cluster
            value = [cluster_map[strategy] for strategy in value]

        changes_numeric = np.diff(value)
        # Convert result of numpy difference into dictionary that maps trial_index -> whether a change occurred (1 or 0)
        change_count = {trial_idx: int(diff_val != 0) for trial_idx, diff_val in enumerate(list(changes_numeric))}
        change_list_of_dicts.append(change_count) # a dict of all changes for each participant, len: 15

    df = pd.DataFrame(change_list_of_dicts)
    sum_values = df.sum(axis=0)

    fig = plt.figure(figsize=(15, 10))
    # create percentages by dividing each item in the list by number of participants (15)
    relative_sum_values = [x / number_participants for x in list(sum_values)]
    if cluster:
        plt.bar(sum_values.keys(), relative_sum_values, 1, color='b')
        plt.ylim(top=1.0)
        plt.xlabel("Trial Number", size=24)
        plt.ylabel("Percentage of people who changed strategy cluster", fontsize=24)
        plt.savefig(f"../results/{exp}_{block}/absolute_number_of_changes_cluster.png",
                    bbox_inches='tight')
    else:
        plt.bar(sum_values.keys(), relative_sum_values, 1, color='b')
        plt.ylim(top=1.0)
        plt.xlabel("Trial Number", size=24)
        plt.ylabel("Percentage of people who changed strategy", fontsize=24)
        plt.savefig(f"../results/{exp}_{block}/absolute_number_of_changes_strategy.png",
                    bbox_inches='tight')
    plt.close(fig)
    return None

def analysis_change_percentage(exp_num, block):
    strategies = learning_utils.pickle_load(f"../results/inferred_strategies/{exp_num}_{block}/strategies.pkl")
    number_participants = 15 #todo: make this more dynamic

    #clusters = learning_utils.pickle_load("data/kl_clusters.pkl")
    cluster_map = learning_utils.pickle_load("data/kl_cluster_map.pkl")

    plot_difference_between_trials(cluster_map, strategies, number_participants, cluster=False)
    plot_difference_between_trials(cluster_map, strategies, number_participants, cluster=True)

    # Get sorted trajectories
    cluster_trajectory, strategy_trajectory = get_sorted_trajectories(cluster_map, strategies)

    # show how many trials until the final strategy was used
    print("Strategy usage:")
    analyze_trajectory(strategy_trajectory, print_trajectories=False)
    print("\n")

    # show how many trials until the final strategy cluster was used
    print("Cluster usage:")
    analyze_trajectory(cluster_trajectory, print_trajectories=False)
    print("\n")

def average_score_development(exp, block, participant_data):
    # plot the average score development
    participant_score = learning_utils.get_participant_scores(exp, participant_data["pid"].tolist())
    participant_score = pd.DataFrame.from_dict(participant_score) #pid as column, trial as row

    # get average score across trials
    participant_mean = participant_score.mean(axis=1)

    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(participant_score.shape[0]), participant_mean)
    plt.ylim(top=50)
    plt.xlabel("Trial Number", size=24)
    plt.ylabel(f"Average score for {exp}", fontsize=24)
    plt.savefig(f"../results/{exp}_{block}/score_development.png",
                bbox_inches='tight')
    plt.close(fig)
    return None


def click_sequence():
    learning_utils.get_clicks()
    return None


if __name__ == "__main__":
    # reward_structure = sys.argv[1]
    # block = None
    # if len(sys.argv) > 2:
    #     block = sys.argv[2]

    reward_structure = "constant_variance"
    block = "training"

    # Load your experiment strategies here as a dict, dict of pid and strategy sequence
    exp_num = reward_structure
    if exp_num == "constant_variance":
        exp = "c1.1"
    elif exp_num == "increasing_variance":
        exp = "v1.0"
    else:
        exp = "c2.1"

    block = "training"
    #analysis_change_percentage(exp_num, block)

    # open participants csv
    data = analysis_utils.get_data(exp)
    participant_data = data['participants']
    average_score_development(exp, block, participant_data)
