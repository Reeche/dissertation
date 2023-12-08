import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm

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
    proportion = sum(value[index] for key, value in mydict.items()) / len(mydict)
    print("The proportion at the 15th trial: ", proportion)
    return proportion

def compare_adaptive_propotion(conditions):
    for condition in conditions:
        data_raw = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        if condition == "mf":
            index = 15
        else:
            index = 0

        # filter data for mf_clicked
        if condition == "mf":
            data = {key: data_raw[key] for key in mf_clicked if key in data_raw}
        else:
            data = data_raw

        proportion_at_index[condition] = calculate_proportion_at_index(replace(data), index)

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
    plt.close()
    return None


def logistic_regression(conditions, mf_clicked, adaptive_strategies):
    strategy_data = pd.DataFrame(columns=["trial", "condition"])
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))
        # filter data for mf_clicked
        if condition == "mf":
            data = data[mf_clicked]

        reshaped_df = pd.melt(data.reset_index(), id_vars='index', var_name='trial', value_name='strategy')
        reshaped_df["condition"] = condition

        # if condition is MF, remove the first 15 trials from the dataframe
        if condition == "mf":
            reshaped_df = reshaped_df[reshaped_df["index"] >= 15]
            reshaped_df["index"] = reshaped_df["index"] - 15

        # if strategy is in adaptive_strategies, replace with 1, else replace with 0
        reshaped_df["strategy"] = reshaped_df["strategy"].apply(lambda x: 1 if x in adaptive_strategies else 0)

        strategy_data = strategy_data._append(reshaped_df)
    strategy_data.columns = ["pid", "condition", "trial", "strategy"]

    # fit logistic regression
    model = sm.GLM.from_formula("strategy ~ C(condition, Treatment('mf')) * trial", data=strategy_data, family=sm.families.Binomial()).fit()
    print(model.summary())
    return None


if __name__ == "__main__":
    adaptive = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84, 45, 11, 6, 7,
                29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    # maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    # others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    mf_clicked = [3, 5, 9, 10, 13, 15, 23, 25, 28, 30, 32, 33, 36, 37, 41, 45, 46, 52, 56, 58, 59, 62, 63, 66, 68,
                  69, 72, 74, 76, 78, 82, 84, 86, 89, 91, 93, 94, 96, 98, 100, 102, 104, 108, 111, 115, 116, 124,
                  125, 126, 127, 129, 130, 132, 134, 137, 138, 139, 141, 145, 146, 148, 149, 152, 156, 158, 159,
                  163, 167, 168, 172, 173, 175, 176, 179, 180, 182, 184, 186, 187, 189, 190, 191]

    logistic_regression(conditions, mf_clicked, adaptive)
    ### Create a contingency table
    proportion_at_index = {}

    mb_mf = np.array([[(1 - proportion_at_index["mb"]) * 100, proportion_at_index["mb"] * 100],
                      [(1 - proportion_at_index["mf"]) * 100, proportion_at_index["mf"] * 100]])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(mb_mf)
    print("Proportion is significantly different between MB and MF at the 15th trial")
    print(p_value)

    mb_stroop = np.array([[(1 - proportion_at_index["mb"]) * 100, proportion_at_index["mb"] * 100],
                          [(1 - proportion_at_index["stroop"]) * 100, proportion_at_index["stroop"] * 100]])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(mb_stroop)
    print("Proportion is significantly different between MB and STROOP at the 15th trial")
    print(p_value)

    stroop_mf = np.array([[(1 - proportion_at_index["mf"]) * 100, proportion_at_index["mf"] * 100],
                          [(1 - proportion_at_index["stroop"]) * 100, proportion_at_index["stroop"] * 100]])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(stroop_mf)
    print("Proportion is significantly different between MF and STROOP at the 15th trial")
    print(p_value)