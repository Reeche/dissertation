from collections import Counter
import matplotlib.pyplot as plt
import operator
import numpy as np
from collections import defaultdict
import sys
from utils import learning_utils, distributions
sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

"""
This file outputs analysis related to convergence in learning. 

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

def get_sorted_trajectories(strategies):
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


def get_changes(input: np.array = None):
    results = {}
    for idx, value in enumerate(input):
        if value == 0:
            results[idx] = 0
        else:
            results[idx] = 1
    return results


def difference_between_trials(strategies: defaultdict, number_participants):
    # returns booleans whether from one trial to the other trial there was a change
    # strateiges is a default dict that contains list of participants (pid) and their strategies across each trial
    change_list_of_dicts = []
    for key, value in strategies.items():
        changes_numeric = np.diff(value)
        changes_boolean_as_dict = get_changes(changes_numeric)
        change_list_of_dicts.append(changes_boolean_as_dict)
    sum_values = sum((Counter(dict(x)) for x in change_list_of_dicts), Counter())

    fig = plt.figure(figsize=(15, 10))
    # create percentages by dividing each item in the list by number of participants (15)
    relative_sum_values = [x / number_participants for x in list(sum_values.values())]
    plt.bar(sum_values.keys(), relative_sum_values, 1, color='b')
    plt.ylim(top=1.0)
    plt.xlabel("Trial Number", size=24)
    plt.ylabel("Percentage of people who changed strategy", fontsize=24)
    plt.savefig(f"../results/{exp}_{block}/absolute_number_of_changes.png",
                bbox_inches='tight')
    plt.close(fig)
    return None


if __name__ == "__main__":
    # reward_structure = sys.argv[1]
    # block = None
    # if len(sys.argv) > 2:
    #     block = sys.argv[2]

    reward_structure = "increasing_variance"
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
    strategies = learning_utils.pickle_load(f"../results/inferred_strategies/{exp_num}_{block}/strategies.pkl")
    number_participants = 15 #todo: make this more dynamic
    difference_between_trials(strategies, number_participants)

    clusters = learning_utils.pickle_load("data/kl_clusters.pkl")
    cluster_map = learning_utils.pickle_load("data/kl_cluster_map.pkl")

    # Get sorted trajectories
    cluster_trajectory, strategy_trajectory = get_sorted_trajectories(strategies)

    # show how many trials until the final strategy was used
    print("Strategy usage:")
    analyze_trajectory(strategy_trajectory)
    print("\n")

    # show how many trials until the final strategy cluster was used
    print("Cluster usage:")
    analyze_trajectory(cluster_trajectory)
    print("\n")