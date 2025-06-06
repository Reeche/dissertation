import ast
import math
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chisquare, wilcoxon

from vars import clicking_pid, learning_pid, planningamount_learners

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
r_stats = importr('stats')

EXPERIMENTS = [
    "high_variance_high_cost",
    "high_variance_low_cost",
    "low_variance_high_cost",
    "low_variance_low_cost"
]

EXPERIMENT_SETTINGS = {
    "high_variance_low_cost": {"click_cost": 0, "variance": 1, "label": "HVLC", "ref_clicks": 9.09, "ref_std": 2.26},
    "high_variance_high_cost": {"click_cost": 1, "variance": 1, "label": "HVHC", "ref_clicks": 6.73, "ref_std": 2.89},
    "low_variance_low_cost": {"click_cost": 0, "variance": 0, "label": "LVLC", "ref_clicks": 3.96, "ref_std": 2.37},
    "low_variance_high_cost": {"click_cost": 1, "variance": 0, "label": "LVHC", "ref_clicks": 0, "ref_std": 0}
}


def create_click_df(data: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Creates a dataframe summarizing clicks from raw data."""
    queries = data["queries"]
    df = pd.DataFrame({
        "pid": data["pid"],
        "trial": data["trial_index"],
        "score": data["score"]
    })

    # Parse clicks from queries
    clicks_list = queries.apply(lambda q: ast.literal_eval(q)["click"]["state"]["target"])
    df["clicks"] = clicks_list
    df["number_of_clicks"] = clicks_list.apply(len)

    settings = EXPERIMENT_SETTINGS.get(experiment, {"click_cost": 0, "variance": 0})
    df["click_cost"] = settings["click_cost"]
    df["variance"] = settings["variance"]
    df["condition"] = experiment
    return df


def calculate_confidence_interval(mean: float, std_dev: float, sample_size: int) -> Tuple[float, float]:
    """Calculate 95% confidence interval for the mean."""
    se = std_dev / math.sqrt(sample_size)
    margin = 1.96 * se
    return mean - margin, mean + margin


def plot_clicks(clicks_df: pd.DataFrame, experiment: str) -> None:
    """Plot average number of clicks by trial with confidence intervals and reference lines."""
    pid_pivot = clicks_df.pivot(index="pid", columns="trial", values="number_of_clicks")
    pid_avg = pid_pivot.mean(axis=0)
    std_dev = pid_pivot.std(axis=0)
    n = pid_pivot.shape[0]
    ci = 1.96 * (std_dev / np.sqrt(n))

    x = np.arange(len(pid_avg))

    plt.plot(pid_avg, label=f"Participants: {pid_avg.iloc[0]:.1f} to {pid_avg.iloc[-1]:.1f} clicks",
             color="blue", linewidth=3)
    plt.fill_between(x, pid_avg - ci, pid_avg + ci, color="blue", alpha=0.1, label="95% CI")

    # Reference lines based on experiment
    settings = EXPERIMENT_SETTINGS.get(experiment)
    if settings:
        plt.axhline(y=settings["ref_clicks"], color='red', linestyle='-')
        lower, upper = calculate_confidence_interval(settings["ref_clicks"], settings["ref_std"], 100_000)
        plt.fill_between(x, lower, upper, color='red', alpha=0.1)

    plt.ylim(-1, 13)
    plt.xlabel("Trial Number", fontsize=14)
    plt.ylabel(f"Average number of clicks ({settings['label'] if settings else experiment})", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
    plt.close()


def plot_individual_clicks(click_df: pd.DataFrame, experiment: str, batch_size: int = 7) -> None:
    """Plot individual click counts by participant in batches."""
    pid_list = click_df["pid"].unique()
    for i in range(0, len(pid_list), batch_size):
        batch = pid_list[i:i + batch_size]
        for pid in batch:
            series = click_df.loc[click_df["pid"] == pid, "number_of_clicks"].to_list()
            plt.plot(series)
        plt.savefig(f"plots/{experiment}_individual_plots_{i // batch_size}.png")
        plt.close()


def trend_test(average_clicks: pd.Series, experiment: str) -> None:
    """Run Mann-Kendall trend test."""
    result = mk.original_test(average_clicks)
    print(f"Mann Kendall test for clicks for {experiment}: {result}")


def normality_test(average_clicks: pd.Series, experiment: str) -> None:
    """Test for normality of clicks using D’Agostino’s K^2 test."""
    k2, p = stats.normaltest(average_clicks)
    print(f"Normality test for clicks for {experiment}: p = {p:.5f}")


def anova(click_data: pd.DataFrame) -> None:
    """Run ANOVA on click data."""
    formula = 'number_of_clicks ~ C(trial) + click_cost + pid + C(variance) + trial:variance + trial:click_cost + trial:variance:click_cost'
    model = ols(formula, data=click_data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    print(table)


def lme(click_data: pd.DataFrame) -> None:
    """Fit linear mixed effects model on click data."""
    formula = "number_of_clicks ~ trial"
    model = smf.mixedlm(formula, data=click_data, groups=click_data["pid"]).fit()
    print(model.summary())


def magnitude_of_change(click_df: pd.DataFrame, experiment: str) -> None:
    """Plot histogram of magnitude of changes in clicks between trials."""
    diffs = click_df.groupby("pid")["number_of_clicks"].apply(lambda x: x.diff().dropna())
    diffs_flat = diffs.explode().astype(int)
    counts = Counter(diffs_flat)
    x, y = zip(*sorted(counts.items()))
    plt.bar(x, y)
    plt.savefig(f"plots/magnitude_{experiment}.png")
    plt.close()


def clicking_pid(click_df: pd.DataFrame, experiment: str) -> List:
    """Return list of pids with at least one click."""
    pid_list = click_df["pid"].unique()
    good_pid = [pid for pid in pid_list if click_df.loc[click_df['pid'] == pid, "number_of_clicks"].any()]
    print(f"{experiment}: {len(good_pid)} participants clicked out of {len(pid_list)}")
    return good_pid


def sequential_dependence(data: pd.DataFrame) -> None:
    """Test for sequential dependence in clicks using Fisher’s exact test."""
    click_df = data[['pid', 'trial', 'number_of_clicks']]
    pivot = click_df.pivot(index="trial", columns="pid", values="number_of_clicks").fillna(0).astype(int)
    pairs = []

    for col in pivot.columns:
        seq = pivot[col].tolist()
        pairs.extend([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])

    max_clicks = pivot.max().max() + 1
    counts_matrix = pd.DataFrame(0, index=range(max_clicks), columns=range(max_clicks))

    for a, b in pairs:
        counts_matrix.at[a, b] += 1

    counts_no_diag = counts_matrix.values[~np.eye(max_clicks, dtype=bool)].reshape(max_clicks, -1)
    res = stats.fisher_test(counts_no_diag, simulate_p_value=True)
    print(res)


def trend_within_participant(experiment: str, data: pd.DataFrame) -> None:
    """Evaluate trend within each participant."""
    click_df = data[['pid', 'trial', 'number_of_clicks']]
    pivot = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")

    sig_improved = []
    improved = []

    for pid in pivot.columns:
        result = mk.original_test(pivot[pid].dropna())
        if experiment in ["high_variance_low_cost", "high_variance_high_cost"]:
            if result.trend == "increasing":
                sig_improved.append(pid)
            if result.s > 0:
                improved.append(pid)
        else:
            if result.trend == "decreasing":
                sig_improved.append(pid)
            if result.s < 0:
                improved.append(pid)

    print(f"Participants with significant improvement in {experiment}: {len(sig_improved)} / {pivot.shape[1]}")
    print(f"Participants with any improvement in {experiment}: {len(improved)} / {pivot.shape[1]}")


def rank_order_changes(click_data: pd.DataFrame) -> None:
    """Calculate and plot changes in participant rank order between first and last trials."""
    ranks = click_data.pivot(index="pid", columns="trial", values="number_of_clicks").rank()
    rank_changes = ranks[1] - ranks[ranks.columns[-1]]

    plt.hist(rank_changes.dropna(), bins=20)
    plt.title("Participant rank order changes from first to last trial")
    plt.show()


if __name__ == "__main__":
    # Example: Load your data (adjust path and format)
    data = pd.read_csv("your_data_file.csv")  # replace with your actual data path

    for experiment in EXPERIMENTS:
        # Create click dataframe for this experiment
        click_df = create_click_df(data, experiment)

        # Basic statistics & plotting
        plot_clicks(click_df, experiment)
        plot_individual_clicks(click_df, experiment)

        # Statistical tests
        average_clicks = click_df.groupby("trial")["number_of_clicks"].mean()
        trend_test(average_clicks, experiment)
        normality_test(average_clicks, experiment)

        # Mixed effects model and ANOVA
        anova(click_df)
        lme(click_df)

        # Other analyses
        magnitude_of_change(click_df, experiment)
        good_pids = clicking_pid(click_df, experiment)

        # Trend within participant
        trend_within_participant(experiment, click_df)

        # Rank order changes
        rank_order_changes(click_df)