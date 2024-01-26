import pandas as pd
import matplotlib.pyplot as plt

"""
Check how much overlap there is between the pid strategy and model planning strategy
Planning strategy is currently not inferred using the CM but using coarse metrics like clicking 1, 5, 9 
constitute forward search

"""


def count_sequence_occurence(sequence):
    # how many of the sublists contain the sequences [1, 5, 9], [1, 5], [1, 9], [5, 9], [1], [5], [9] in any order
    count = 0
    for sublist in sequence:
        if set(sublist) == set([1, 5, 9, 0]):
            count += 1
        elif set(sublist) == set([1, 5, 0]):
            count += 1
        elif set(sublist) == set([1, 9, 0]):
            count += 1
        elif set(sublist) == set([5, 9, 0]):
            count += 1
        elif set(sublist) == set([1, 0]):
            count += 1
        elif set(sublist) == set([5, 0]):
            count += 1
        elif set(sublist) == set([9, 0]):
            count += 1

    # print(len(sequence))
    # print(count)
    # print(count / len(sequence))
    return count


def trialwise_overlap(model_clicks, pid_clicks):
    ## how many clck sequences overlap between model and pid for each trial?
    data = pd.DataFrame({"model_clicks": model_clicks, "pid_clicks": pid_clicks})

    overlap_all = []
    for model, pid in zip(data["model_clicks"], data["pid_clicks"]):
        overlap = []
        for model_, pid_ in zip(model, pid):
            if set(model_) == set(pid_):
                overlap.append(1)
            else:
                overlap.append(0)
        overlap_all.append(overlap)

    data["overlap"] = overlap_all

    # convert data["overlap"] column to array
    df_expanded = data["overlap"].apply(pd.Series)
    sum_across_index = df_expanded.sum()

    # proportion
    sum_across_index = sum_across_index / len(data)

    # plot trialwise_overlap
    plt.plot(sum_across_index)
    plt.show()
    plt.close()



if __name__ == "__main__":
    data = pd.read_csv(f"../likelihood_vanilla_model_comparison/data/strategy_discovery.csv", index_col=0)

    # filter for model = 1756
    data = data[data["model"] == 1756]

    model_clicks = data["model_clicks"]
    pid_clicks = data["pid_clicks"]

    ### convert from string to list
    model_clicks = [eval(x) for x in model_clicks]
    pid_clicks = [eval(x) for x in pid_clicks]

    # trialwise_overlap(model_clicks, pid_clicks)

    ### get trial by trial count of 1,5,9
    trial_count = {key: 0 for key in range(0, 120)}
    for clicks in model_clicks:
        for trial, sequence in enumerate(clicks):
            # if sequence in set 1, 5 or 1, 9
            if set(sequence) == set([1, 5, 9, 0]):
                trial_count[trial] += 1

    print(trial_count)
    plt.plot(trial_count.values())
    plt.show()
    plt.close()

    ### flatten the lists to get total count of overlapping sequences
    # model_clicks = [item for sublist in model_clicks for item in sublist]
    # pid_clicks = [item for sublist in pid_clicks for item in sublist]
    #
    # count_sequence_occurence(model_clicks)
    # count_sequence_occurence(pid_clicks)
