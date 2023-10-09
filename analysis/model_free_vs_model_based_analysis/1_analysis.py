import pandas as pd
import matplotlib.pyplot as plt

"""
Analyse whether 
1. pid in mb condition start with a higher proportion of adaptive strategy than pid in the mf condition
2. pid in MF and MB condition start with a higher proportion of adaptive strategy than pid in the stroop condition

For this, need to fit the CM to all participants 
30 trials for MF condition
15 trials for each MB and stroop condition
"""


def replace(mydict):
    replaced_dict = {}
    # replace strategy with 1 for adaptive and 0 for anything else
    for key, value in mydict.items():
        new_value = [1 if x in adaptive else 0 for x in value]
        replaced_dict[key] = new_value
    return replaced_dict


def merge(mb, mf, stroop):
    merged_dict = {}

    for key, value in mb.items():
        merged_dict[f"{key}_mb"] = value

    for key, value in mf.items():
        merged_dict[f"{key}_mf"] = value

    for key, value in stroop.items():
        merged_dict[f"{key}_stroop"] = value

    return merged_dict


def calculate_proportion_at_index(mydict, index):
    # calculate the proportion of adaptive strategies at index 0 for MB and stroop; at index 15 for MF
    proportion = sum(value[index] for key,value in mydict.items()) / len(mydict)
    print(proportion)

if __name__ == "__main__":
    adaptive = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84, 45, 11, 6, 7,
                29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    # maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    # others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    for condition in conditions:
        data = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        if condition == "mf":
            index = 15
        else:
            index = 0
        calculate_proportion_at_index(replace(data),  index)

        # Step 1: Calculate the proportion of 1s for each dictionary
        proportions_dict1 = [sum(values) / len(values) for values in zip(*replace(data).values())]
        if condition == "mf":
            x_values = list(range(30))
            plt.plot(x_values, proportions_dict1, label="MF")
        else:
            x_values = list(range(15, 30))
            plt.plot(x_values, proportions_dict1, label=condition.upper())
    plt.xlabel("Trial")
    plt.ylabel("Proportion of adaptive strategies")
    plt.legend()
    plt.show()

