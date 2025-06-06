import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from vars import clicked_dict

"""
Analysis Goals:
1. Determine whether participants in the MB condition begin with a higher proportion 
   of adaptive strategies than those in the MF condition.
2. Assess whether participants in both MF and MB conditions begin with a higher 
   proportion of adaptive strategies compared to those in the Stroop condition.

Trial Setup:
- MF condition: 30 trials
- MB and Stroop conditions: 15 trials each
"""


def replace_with_binary(mydict, adaptive_strategies):
    """
    Replace strategy codes with 1 if adaptive, otherwise 0.

    Parameters:
        mydict (dict): Dictionary with participant IDs as keys and strategy code lists as values.
        adaptive_strategies (list): List of codes classified as adaptive strategies.

    Returns:
        dict: Dictionary with binary values (1 = adaptive, 0 = non-adaptive).
    """
    return {key: [1 if x in adaptive_strategies else 0 for x in value] for key, value in mydict.items()}


def merge_conditions(mb, mf, stroop):
    """
    Merge condition dictionaries into a single dictionary with tagged keys.

    Parameters:
        mb (dict): Model-based strategies.
        mf (dict): Model-free strategies.
        stroop (dict): Stroop task strategies.

    Returns:
        dict: Merged dictionary with keys suffixed by condition.
    """
    return {
        **{f"{key}_mb": val for key, val in mb.items()},
        **{f"{key}_mf": val for key, val in mf.items()},
        **{f"{key}_stroop": val for key, val in stroop.items()}
    }


def calculate_proportion_at_index(mydict, index):
    """
    Calculate the proportion of adaptive strategies at a specific trial index.

    Parameters:
        mydict (dict): Dictionary with binary-encoded strategy lists.
        index (int): Trial index to evaluate.

    Returns:
        float: Proportion of adaptive strategies.
    """
    proportion = sum(value[index] for value in mydict.values()) / len(mydict)
    print("The proportion at the trial:", proportion)
    return proportion


def count_adaptive_strategies(mydict, index):
    """
    Count the number of adaptive strategies at a given index.

    Parameters:
        mydict (dict): Dictionary with binary-encoded strategy lists.
        index (int): Trial index to count at.

    Returns:
        int: Count of adaptive strategies.
    """
    return sum(value[index] for value in mydict.values())


def plot_adaptive_proportion(conditions, clicked_df):
    """
    Plot the proportion of adaptive strategies over trials for each condition.

    Parameters:
        conditions (list): List of condition names.
        clicked_df (dict): Dictionary of participant IDs per condition.
    """
    for condition in conditions:
        data_raw = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        index_start = 14 if condition == "mf" else 0

        # Filter clicked participants
        filtered_data = {k: v for k, v in data_raw.items() if k in clicked_df[condition]}
        binary_data = replace_with_binary(filtered_data, adaptive_strategies)

        # Compute average proportion across participants per trial
        proportions = [sum(vals) / len(vals) for vals in zip(*binary_data.values())]
        std_error = np.std(proportions, ddof=1) / np.sqrt(len(proportions))
        margin_of_error = 1.96 * std_error

        if condition == "mf":
            x_vals = range(30)
            label = "Hybrid"
        else:
            x_vals = range(15, 30)
            label = "Model-based" if condition == "mb" else "Stroop"

        plt.plot(x_vals, proportions, label=label)
        plt.fill_between(x_vals,
                         np.array(proportions) + margin_of_error,
                         np.array(proportions) - margin_of_error,
                         alpha=0.2)

    plt.xlabel("Trial")
    plt.ylabel("Proportion of Adaptive Strategies")
    plt.legend()
    plt.savefig("plots/proportion_at_index.png")
    plt.close()


def logistic_regression(conditions, clicked_dict, adaptive_strategies):
    """
    Run logistic regression to assess the interaction between trial number and condition.

    Parameters:
        conditions (list): Experimental conditions.
        clicked_dict (dict): Dictionary of participant IDs per condition.
        adaptive_strategies (list): List of adaptive strategy codes.
    """
    strategy_data = pd.DataFrame(columns=["trial", "condition"])

    for condition in conditions:
        raw_data = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        data = pd.DataFrame.from_dict(raw_data)
        selected_columns = [col for col in clicked_dict[condition] if col in data.columns]
        selected_df = data[selected_columns]

        reshaped = pd.melt(selected_df.reset_index(), id_vars="index", var_name="trial", value_name="strategy")
        reshaped["condition"] = condition

        if condition == "mf":
            reshaped["index"] = reshaped["index"] - 14

        reshaped["strategy"] = reshaped["strategy"].apply(lambda x: 1 if x in adaptive_strategies else 0)
        strategy_data = pd.concat([strategy_data, reshaped])

    strategy_data.columns = ["pid", "condition", "trial", "strategy"]

    model = sm.GLM.from_formula("strategy ~ C(condition, Treatment('mf')) * trial",
                                data=strategy_data,
                                family=sm.families.Binomial()).fit()
    print(model.summary())


def chi_test(conditions, clicked_dict, adaptive_strategies):
    """
    Perform chi-square tests on adaptive strategy proportions at the last trial.

    Parameters:
        conditions (list): Experimental conditions.
        clicked_dict (dict): Dictionary of participant IDs per condition.
        adaptive_strategies (list): List of adaptive strategy codes.
    """
    count_at_index = {}

    for condition in conditions:
        raw_data = pd.read_pickle(f"../../results/cm/inferred_strategies/{condition}_training/strategies.pkl")
        trial_index = 29 if condition == "mf" else 14
        filtered_data = {k: v for k, v in raw_data.items() if k in clicked_dict[condition]}
        binary_data = replace_with_binary(filtered_data, adaptive_strategies)
        calculate_proportion_at_index(binary_data, trial_index)
        count_at_index[condition] = count_adaptive_strategies(binary_data, trial_index)

    # Contingency tables
    mb_mf = np.array([
        [count_at_index["mf"], count_at_index["mb"]],
        [len(clicked_dict["mf"]) - count_at_index["mf"], len(clicked_dict["mb"]) - count_at_index["mb"]],
    ])
    print("MB vs MF:")
    print(stats.chi2_contingency(mb_mf))

    mb_stroop = np.array([
        [count_at_index["mb"], count_at_index["stroop"]],
        [len(clicked_dict["mb"]) - count_at_index["mb"], len(clicked_dict["stroop"]) - count_at_index["stroop"]],
    ])
    print("MB vs Stroop:")
    print(stats.chi2_contingency(mb_stroop))

    mf_stroop = np.array([
        [count_at_index["mf"], count_at_index["stroop"]],
        [len(clicked_dict["mf"]) - count_at_index["mf"], len(clicked_dict["stroop"]) - count_at_index["stroop"]],
    ])
    print("MF vs Stroop:")
    print(stats.chi2_contingency(mf_stroop))

if __name__ == "__main__":
    adaptive_strategies = [65, 64, 24, 21, 63, 43, 17, 16, 57, 59, 88, 54, 4, 31, 26, 82, 37, 48, 50, 85, 76, 18, 84,
                           45, 11, 6, 7,
                           29, 80, 36, 67, 12, 87, 60, 5, 10, 49, 42, 72, 51, 2, 71, 13, 40, 14, 15, 78, 56]
    maladaptive = [22, 53, 39, 28, 30, 23, 66, 70, 74]
    others = [33, 44, 27, 79, 69, 34, 61, 73, 32]
    conditions = ["mf", "mb", "stroop"]

    plot_adaptive_proportion(conditions, clicked_dict)
    logistic_regression(conditions, clicked_dict, adaptive_strategies)
    chi_test(conditions, clicked_dict, adaptive_strategies)
