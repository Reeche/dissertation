import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from vars import clicked_dict

"""
Analyse whether 
1. pid in mb condition start with a higher proportion of adaptive strategy than pid in the mf condition
2. pid in MF and MB condition start with a higher proportion of adaptive strategy than pid in the stroop condition

For this, need to fit the CM to all participants 
30 trials for MF condition
15 trials for each MB and stroop condition
"""


def replace(mydict, adaptive_strategies):
    replaced_dict = {}
    # replace strategy with 1 for adaptive and 0 for anything else
    for key, value in mydict.items():
        new_value = [1 if x in adaptive_strategies else 0 for x in value]
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


def count_adaptive_strategies(mydict, index):
    return sum(value[index] for key, value in mydict.items())


def plot_adaptive_propotion(conditions, clicked_df):
    # proportion_at_index = {}
    for condition in conditions:
        data_raw = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        if condition == "mf":
            index = 14
        else:
            index = 0

        # filter data for clicked
        data = {key: data_raw[key] for key in clicked_df[condition] if key in data_raw}

        # what is this for?
        # proportion_at_index[condition] = calculate_proportion_at_index(replace(data, adaptive_strategies), index)

        # Step 1: Calculate the proportion of 1s for each dictionary
        proportions_list = [sum(values) / len(values) for values in zip(*replace(data, adaptive_strategies).values())]
        if condition == "mf":
            x_values = list(range(30))
            plt.plot(x_values, proportions_list, label="Hybrid")
            # CI
            std_error = np.std(proportions_list, ddof=1) / np.sqrt(len(proportions_list))
            margin_of_error = 1.96 * std_error  # 1.96 is the z-value for 95% confidence interval
            plt.fill_between(x_values, proportions_list + margin_of_error, proportions_list - margin_of_error,
                             alpha=0.2)


        else:
            x_values = list(range(15, 30))
            if condition == "mb":
                plt.plot(x_values, proportions_list, label="Model-based")
            else:
                plt.plot(x_values, proportions_list, label="Stroop")
            # CI
            std_error = np.std(proportions_list, ddof=1) / np.sqrt(len(proportions_list))
            margin_of_error = 1.96 * std_error  # 1.96 is the z-value for 95% confidence interval
            plt.fill_between(x_values, proportions_list + margin_of_error, proportions_list - margin_of_error,
                             alpha=0.2)

    plt.xlabel("Trial")
    plt.ylabel("Proportion of adaptive strategies")
    plt.legend()
    # plt.show()
    plt.savefig(f"plots/proportion_at_index.png")
    plt.close()
    return None


def logistic_regression(conditions, clicked_dict, adaptive_strategies):
    strategy_data = pd.DataFrame(columns=["trial", "condition"])
    for condition in conditions:
        data = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl"))
        # select only columns in data that are in clicked_dict
        selected_columns = clicked_dict[condition]
        selected_columns = [col for col in selected_columns if
                            col in data.columns]  # convert to the same type as header
        selected_df = data[selected_columns]

        reshaped_df = pd.melt(selected_df.reset_index(), id_vars='index', var_name='trial', value_name='strategy')
        reshaped_df["condition"] = condition

        # if condition is MF, remove the first 15 trials from the dataframe
        # todo: if we want to compare FIRST 15 trials of MF with the 15 first: [reshaped_df["index"] <= 15]
        if condition == "mf":
            # reshaped_df = reshaped_df[reshaped_df["index"] <= 14]
            reshaped_df["index"] = reshaped_df["index"] - 14

        # if strategy is in adaptive_strategies, replace with 1, else replace with 0
        reshaped_df["strategy"] = reshaped_df["strategy"].apply(lambda x: 1 if x in adaptive_strategies else 0)

        strategy_data = strategy_data._append(reshaped_df)
    strategy_data.columns = ["pid", "condition", "trial", "strategy"]

    # fit logistic regression
    model = sm.GLM.from_formula("strategy ~ C(condition, Treatment('mf')) * trial", data=strategy_data,
                                family=sm.families.Binomial()).fit()
    print(model.summary())
    return None


def chi_test(conditions, clicked_dict, adaptive_strategies):
    ### Create a contingency table
    proportion_at_index = {}
    count_at_index = {}

    for condition in conditions:
        data_raw = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        if condition == "mf":
            # index = 14 #first test trial
            index = 29  # last trial
        else:
            # index = 0 #first test trial
            index = 14  # last trial

        # filter data for mf_clicked
        data = {key: data_raw[key] for key in clicked_dict[condition] if key in data_raw}

        proportion_at_index[condition] = calculate_proportion_at_index(replace(data, adaptive_strategies), index)
        count_at_index[condition] = count_adaptive_strategies(replace(data, adaptive_strategies), index)

    sum_mf_mb = count_at_index["mf"] + count_at_index["mb"]
    sum_mf_stroop = count_at_index["mf"] + count_at_index["stroop"]
    sum_mb_stroop = count_at_index["mb"] + count_at_index["stroop"]

    mb_mf = np.array([[(sum_mf_mb - count_at_index["mb"]), count_at_index["mb"]],
                      [(sum_mf_mb - count_at_index["mf"]), count_at_index["mf"]]])
    print("Proportion is significantly different between MB and MF at the 15th trial")
    print(stats.chi2_contingency(mb_mf))

    mb_stroop = np.array([[sum_mb_stroop - count_at_index["mb"], count_at_index["mb"]],
                          [sum_mb_stroop - count_at_index["stroop"], count_at_index["stroop"]]])
    print("Proportion is significantly different between MB and STROOP at the 15th trial")
    print(stats.chi2_contingency(mb_stroop))

    stroop_mf = np.array([[sum_mf_stroop - count_at_index["mf"], count_at_index["mf"]],
                          [sum_mf_stroop - count_at_index["stroop"], count_at_index["stroop"]]])
    # chi2_stat, p_value, dof, expected = stats.chi2_contingency(stroop_mf)
    print("Proportion is significantly different between MF and STROOP at the 15th trial")
    print(stats.chi2_contingency(stroop_mf))


if __name__ == "__main__":
    adaptive_strategies = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84,
                           45, 11, 6, 7,
                           29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    # maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    # others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    plot_adaptive_propotion(conditions, clicked_dict)
    # logistic_regression(conditions, clicked_dict, adaptive_strategies)
    # chi_test(conditions, clicked_dict, adaptive_strategies)
