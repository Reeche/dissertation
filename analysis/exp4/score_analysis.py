import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from vars import clicked_dict


def plot_with_confidence_interval(x, y, sem, label):
    """
    Helper to plot mean ± 95% CI.
    """
    plt.plot(x, y, label=label)
    ci_upper = y + 1.96 * sem
    ci_lower = y - 1.96 * sem
    plt.fill_between(x, ci_upper, ci_lower, alpha=0.2)


def plot_actual_score(conditions, clicked):
    """
    Plots actual score over trials for each condition, including confidence intervals.
    """
    for cond in conditions:
        df = pd.read_csv(f"../../data/human/{cond}/mouselab-mdp.csv")
        df = df[df["block"] == "training"]
        df = df[df["pid"].isin(clicked[cond])]

        df["trial_index"] = df.groupby("pid").cumcount() + 1
        df = df[["pid", "trial_index", "score"]]

        mean_scores = df.groupby("trial_index")["score"].mean()
        se_scores = df.groupby("trial_index")["score"].sem()

        if cond == "mf":
            x = list(range(30))
            plot_with_confidence_interval(x, mean_scores, se_scores, label="Hybrid")
        else:
            x = list(range(15, 30))
            label = "Model-based" if cond == "mb" else "Stroop"
            plot_with_confidence_interval(x, mean_scores[15:], se_scores[15:], label=label)

    plt.xlabel("Trial")
    plt.ylabel("Actual score")
    plt.legend()
    plt.title("Actual Score Over Trials")
    plt.show()
    plt.close()


def plot_expected_score(conditions, clicked):
    """
    Plots expected score based on inferred strategy scores over trials.
    """
    score_mapping = pd.read_pickle("../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")

    for cond in conditions:
        strategy_df = pd.read_pickle(f"../../results/cm/inferred_strategies/{cond}_training/strategies.pkl")
        strategy_df = pd.DataFrame.from_dict(strategy_df)
        selected_cols = [col for col in clicked[cond] if col in strategy_df.columns]
        trial_data = strategy_df[selected_cols] - 1  # adjust index
        trial_data = trial_data.replace(score_mapping)

        mean_scores = trial_data.mean(axis=1)
        se_scores = trial_data.sem(axis=1)

        if cond == "mf":
            x = list(range(30))
            plot_with_confidence_interval(x, mean_scores, se_scores, label="Hybrid")
        else:
            x = list(range(15, 30))
            label = "Model-based" if cond == "mb" else "Stroop"
            plot_with_confidence_interval(x, mean_scores[15:], se_scores[15:], label=label)

    plt.xlabel("Trial")
    plt.ylabel("Expected score")
    plt.legend()
    plt.title("Expected Score Over Trials")
    plt.show()
    plt.close()


def compare_expected_score(conditions, clicked):
    """
    Compares expected scores statistically using mixed effects and Mann-Whitney U tests.
    """
    score_mapping = pd.read_pickle("../../results/cm/strategy_scores/v1.0_strategy_scores.pkl")
    df_all = []

    for cond in conditions:
        strategy_df = pd.read_pickle(f"../../results/cm/inferred_strategies/{cond}_training/strategies.pkl")
        strategy_df = pd.DataFrame.from_dict(strategy_df)
        selected_cols = [col for col in clicked[cond] if col in strategy_df.columns]
        trial_data = strategy_df[selected_cols] - 1
        trial_data = trial_data.replace(score_mapping)

        long_df = trial_data.reset_index().melt(id_vars='index', var_name='pid', value_name='score')
        long_df['condition'] = cond

        if cond == "mf":
            long_df = long_df[long_df["index"] >= 14]
            long_df["index"] -= 14

        df_all.append(long_df)

    df_final = pd.concat(df_all, ignore_index=True)
    df_final.columns = ["trial", "pid", "score", "condition"]

    regression_analysis(df_final)
    mann_whitney_test(df_final)


def compare_actual_score(conditions, clicked):
    """
    Compares actual scores statistically using mixed effects and Mann-Whitney U tests.
    """
    df_all = []

    for cond in conditions:
        df = pd.read_csv(f"../../data/human/{cond}/mouselab-mdp.csv")
        df = df[df["pid"].isin(clicked[cond])]
        df["trial_index"] = df.groupby("pid").cumcount() + 1

        if cond == "mf":
            df = df[df["trial_index"] >= 15]
            df["trial_index"] -= 14

        df["condition"] = cond
        df = df[["pid", "score", "trial_index", "condition"]]
        df.columns = ["pid", "score", "trial", "condition"]
        df_all.append(df)

    df_final = pd.concat(df_all, ignore_index=True)

    regression_analysis(df_final)
    mann_whitney_test(df_final)


def regression_analysis(df):
    """
    Runs mixed-effects model with trial and condition interaction.
    """
    model = mixedlm(
        "score ~ trial * C(condition, Treatment('mf'))",
        data=df,
        groups=df["pid"]
    ).fit()
    print(model.summary())


def mann_whitney_test(df):
    """
    Runs Mann-Whitney and Kruskal-Wallis tests on the first trial across conditions.
    """
    df_trial1 = df[df["trial"] == 1]

    for cond in ["mf", "mb", "stroop"]:
        subset = df_trial1[df_trial1["condition"] == cond]
        print(f"{cond.upper()} Mean: {subset['score'].mean():.2f}, SD: {subset['score'].std():.2f}")

    print("\nKruskal-Wallis test across all conditions:")
    kw_result = stats.kruskal(
        *(df_trial1[df_trial1["condition"] == cond]["score"] for cond in ["mf", "mb", "stroop"])
    )
    print(kw_result)

    print("\nMann-Whitney U tests:")
    print("MF vs MB:", stats.mannwhitneyu(
        df_trial1[df_trial1["condition"] == "mf"]["score"],
        df_trial1[df_trial1["condition"] == "mb"]["score"],
        alternative="greater"
    ))

    print("MF vs STROOP:", stats.mannwhitneyu(
        df_trial1[df_trial1["condition"] == "mf"]["score"],
        df_trial1[df_trial1["condition"] == "stroop"]["score"],
        alternative="greater"
    ))

    print("MB vs STROOP:", stats.mannwhitneyu(
        df_trial1[df_trial1["condition"] == "mb"]["score"],
        df_trial1[df_trial1["condition"] == "stroop"]["score"],
        alternative="two-sided"
    ))


if __name__ == "__main__":
    conditions = ["mf", "mb", "stroop"]

    # Use one of the following to run the desired analysis
    plot_actual_score(conditions, clicked_dict)
    compare_actual_score(conditions, clicked_dict)

    plot_expected_score(conditions, clicked_dict)
    compare_expected_score(conditions, clicked_dict)
