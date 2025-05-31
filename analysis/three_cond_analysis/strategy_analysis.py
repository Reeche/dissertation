import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pymannkendall as mk
from sklearn.cluster import KMeans
from scipy.stats import chisquare
from collections import Counter

from vars import clicking_pid, assign_model_names, threecond_learners, strategy_mapping

# surpress warnings
import warnings

warnings.filterwarnings("ignore")


### ---------------------- CLUSTERING AND CLASSIFICATION ---------------------- ###

def clustering_kmeans(strategy_scores):
    kmeans = KMeans(n_clusters=3, n_init="auto", max_iter=10000, random_state=42)
    kmeans.fit(strategy_scores.values)
    strategy_scores["label"] = kmeans.labels_

    centers = pd.Series(kmeans.cluster_centers_.flatten()).sort_values()
    strategy_scores["label"] = strategy_scores["label"].replace({
        int(centers.index[0]): "mal",
        int(centers.index[1]): "mod",
        int(centers.index[2]): "adaptive"
    })
    return strategy_scores


def classify_strategies(used_strategies, experiment):
    if experiment not in strategy_mapping:
        raise ValueError(f"Unknown experiment version: {experiment}")

    mapping_dict = strategy_mapping[experiment]

    for label, values in mapping_dict.items():
        used_strategies.replace({col: {v: label for v in values} for col in used_strategies.columns}, inplace=True)

    return used_strategies


### ---------------------- STRATEGY ANALYSIS ---------------------- ###

def trend_test(data):
    for col in data.columns:
        result = mk.original_test(data[col])
        print(col, result)


def logistic_regression(df, mapping_dict):
    df["strategy"] = [1 if x == "adaptive" else 0 for x in df["strategy"]]
    for condition in df["condition"].unique():
        model = sm.GLM.from_formula("strategy ~ trial", data=df[df["condition"] == condition],
                                    family=sm.families.Binomial()).fit()
        print(condition)
        print(model.summary())


def frequency(kmeans_results):
    counter = Counter(kmeans_results.labels_)
    print("Simulation")
    for k, v in counter.items():
        print(k, v / len(kmeans_results.labels_))


def compare_last_trial_proportions(participants):
    last_strategy = [v[-1] for v in participants.values() if isinstance(v, list) and len(v) > 0]
    mapped = list(map(mapping_dict.get, last_strategy))
    counter = Counter(mapped)
    print("Participants")
    for k in counter:
        print(k, counter[k] / len(mapped))


def adaptive_proportion_higher_than_chance(strategy_labels, participants_df):
    cm_counts = strategy_labels["label"].value_counts(normalize=True) * 100
    pid_counts = participants_df.iloc[:, -1:].replace(mapping_dict).value_counts(normalize=True) * 100

    non_adaptive_cm = cm_counts.get("mal", 0) + cm_counts.get("mod", 0)
    non_adaptive_pid = pid_counts.get("mal", 0) + pid_counts.get("mod", 0)

    print("pid", pid_counts.get("adaptive", 0), non_adaptive_pid)
    print("CM", cm_counts.get("adaptive", 0), non_adaptive_cm)
    res = chisquare([pid_counts.get("adaptive", 0), non_adaptive_pid],
                    f_exp=[cm_counts.get("adaptive", 0), non_adaptive_cm])
    print(res)


### ---------------------- PLOTTING FUNCTIONS ---------------------- ###

def plot_all_strategy_proportions(data, experiment, plot=True):
    # Map numeric strategy codes to labels using classify_strategies
    df = classify_strategies(data.copy(), experiment)

    frequencies = pd.DataFrame(columns=["Adaptive", "Mod. adaptive", "Maladaptive"])
    counts_df = pd.DataFrame(columns=["Adaptive", "Mod. adaptive", "Maladaptive"])

    for col in df:
        counts = Counter(df[col])
        row = {
            'Adaptive': counts.get("Adaptive", 0) / len(df),
            'Mod. adaptive': counts.get("Mod. adaptive", 0) / len(df),
            'Maladaptive': counts.get("Maladaptive", 0) / len(df)
        }
        frequencies = frequencies._append(row, ignore_index=True)

        # absolute counts
        counts = Counter(df[col])
        row_counts = {
            'Adaptive': counts.get("Adaptive", 0),
            'Mod. adaptive': counts.get("Mod. adaptive", 0),
            'Maladaptive': counts.get("Maladaptive", 0)
        }
        counts_df = counts_df._append(row_counts, ignore_index=True)

    trend_test(frequencies)
    frequencies *= 100

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies,
                 label=[f"{k}, {frequencies[k].iloc[0]:.2f}% to {frequencies[k].iloc[-1]:.2f}%" for k in frequencies])
        error_margin = 1.96 * (frequencies.std() / np.sqrt(len(frequencies)))
        for col in frequencies:
            plt.fill_between(frequencies.index, frequencies[col] - error_margin[col],
                             frequencies[col] + error_margin[col], alpha=0.2)
        plt.ylim(-10, 100)
        plt.xlabel("Trials", fontsize=14)
        plt.ylabel("Proportion", fontsize=14)
        plt.legend(fontsize=14)
        plt.show()
        plt.close()
    return frequencies, counts_df


def calculate_and_plot(data, model_names, participants, experiment, pid_mapping, strategy_type="Adaptive"):
    x = np.arange(0, 35)
    for model_name in model_names:
        model_df = data[data["model"] == model_name]
        temp = pd.DataFrame(model_df['model_strategies'].tolist(), columns=[f'{i + 1}' for i in range(35)])
        temp = classify_strategies(temp, experiment)  # <- strategy_mapping used here
        proportions = temp.apply(lambda col: col.value_counts(normalize=True).get(strategy_type, 0), axis=0)
        proportions = proportions * 100
        label = f"{model_name}, {proportions.iloc[0]:.2f}% to {proportions.iloc[-1]:.2f}%"
        plt.plot(x, proportions, label=label)

    participants_frequencies, participants_counts = plot_all_strategy_proportions(participants.copy(), experiment,
                                                                                  plot=False)
    stderr = participants_frequencies.std() / np.sqrt(len(participants_frequencies))
    conf_interval = 1.96 * stderr
    plt.plot(participants_frequencies[strategy_type], label="Participant", color="blue", linewidth=3)
    plt.fill_between(np.arange(0, len(participants_frequencies)),
                     participants_frequencies[strategy_type] - conf_interval[strategy_type],
                     participants_frequencies[strategy_type] + conf_interval[strategy_type],
                     color='blue', alpha=0.1, label=f'95% CI {strategy_type}')
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel(f"Proportion of {strategy_type} strategies", fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=10, ncol=1)
    plt.show()
    plt.close()


def plot_adaptive_proportion(data, participants, experiment):
    data['model'] = data.apply(assign_model_names, axis=1)
    data = data[["model", "model_strategies", "pid_strategies"]]
    data['model_strategies'] = data['model_strategies'].apply(ast.literal_eval)
    data['pid_strategies'] = data['pid_strategies'].apply(ast.literal_eval)

    model_groups = {
        "Alternatives": ["Non-learning", "SSL", "Habitual"],
        "MF": ["hybrid LVOC", "hybrid Reinforce", "MF - LVOC", "MF - Reinforce"],
        "MB": ["MB - Uniform, grouped", "MB - Uniform, ind.",
               "MB - Equal, ind.", "MB - Equal, grouped",
               "MB - Level, grouped", "MB - Level, ind."]
    }

    # Convert pid_mapping (strategy DataFrame) to labels
    pid_mapping = classify_strategies(data, experiment)

    for models in model_groups.values():
        calculate_and_plot(data, models, participants, experiment, pid_mapping)


### ---------------------- FEEDBACK ANALYSIS ---------------------- ###

def changed_strategy_after_negative_feedback(participants_df, exp):
    score_df = pd.read_csv(f"../../data/human/{exp}/mouselab-mdp.csv")
    participants_df = participants_df[participants_df.index.isin(clicking_pid[exp])]
    score_df = score_df[score_df["pid"].isin(participants_df.index)]

    melted = pd.melt(participants_df.T.reset_index(), id_vars="index")
    melted.columns = ["trial", "pid", "strategy"]
    melted["trial_index"] = melted.groupby("pid").cumcount() + 1

    merged = melted.merge(score_df[["pid", "trial_index", "score"]], on=["pid", "trial_index"])
    merged["negative_feedback"] = merged["score"] < 0
    merged["changed_strategy"] = merged.groupby("pid")["strategy"].shift(1) != merged["strategy"]
    merged["changed_after_negative_feedback"] = merged["changed_strategy"] & merged["negative_feedback"]

    print(f"Total changed strategy: {merged['changed_strategy'].sum()}")
    print(f"Total changed after negative: {merged['changed_after_negative_feedback'].sum()}")
    pct = (merged['changed_after_negative_feedback'].sum() / merged['changed_strategy'].sum()) * 100
    print(f"Percentage changed after negative: {pct:.2f}%")
    print(
        f"Participants who changed: {merged.groupby('pid')['changed_after_negative_feedback'].sum().gt(0).sum()} / {merged['pid'].nunique()}")


def changed_expected_score_after_negative_feedback(exp):
    score_df = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)
    score_df = score_df.drop_duplicates(subset=["pid"])
    score_df = score_df[score_df["pid"].isin(clicking_pid[exp])]
    score_df["pid_mer"] = score_df["pid_mer"].apply(ast.literal_eval)
    score_df = score_df.explode('pid_mer').reset_index()
    score_df["trial_index"] = score_df.groupby("pid").cumcount() + 1
    score_df["negative_feedback"] = score_df["pid_mer"] < 0
    score_df["changed_score"] = score_df.groupby("pid")["pid_mer"].shift(1) != score_df["pid_mer"]
    score_df["changed_after_negative_feedback"] = score_df["changed_score"] & score_df["negative_feedback"]

    print(f"Changed expected scores: {score_df['changed_score'].sum()}")
    print(f"Changed after negative: {score_df['changed_after_negative_feedback'].sum()}")
    changed_count = score_df.groupby("pid")["changed_after_negative_feedback"].sum().gt(0).sum()
    print(
        f"Participants who changed after negative feedback: {changed_count}/{score_df['pid'].nunique()} ({changed_count / score_df['pid'].nunique():.2%})")


### ---------------------- MAIN EXECUTION ---------------------- ###

if __name__ == "__main__":
    experiments = ["c1.1"]
    for experiment in experiments:
        print(f"Experiment: {experiment}")
        strategy_scores = pd.read_pickle(f"../../results/strategy_scores/{experiment}_strategy_scores.pkl")
        strategy_scores = pd.DataFrame.from_dict(strategy_scores, orient='index')
        strategy_scores.index += 1

        participants = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")
        participants = {k: v for k, v in participants.items() if k in clicking_pid[experiment]}
        participants_df = pd.DataFrame.from_dict(participants, orient='index')

        changed_strategy_after_negative_feedback(participants_df, experiment)
        changed_expected_score_after_negative_feedback(experiment)

        used_strategies = pd.unique(participants_df.values.flatten())
        used_strategy_scores = strategy_scores.loc[used_strategies]
        strategy_labels = clustering_kmeans(used_strategy_scores)
        mapping = used_strategy_scores.set_index(strategy_labels.index)['label']

        # plot_all_strategy_proportions(participants_df, experiment)

        ### load CM model data
        model_data = pd.read_csv(f"../../final_results/model_cm_300_fit/{experiment}.csv")
        ### filter for clicking participants
        model_data = model_data[model_data["pid"].isin(clicking_pid[experiment])]
        plot_adaptive_proportion(model_data, participants_df, experiment)
