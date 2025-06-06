import matplotlib.pyplot as plt
import pandas as pd
from vars import (habitual_pid, mf_pid, hybrid_pid, habitual_not_examined_all_pid, habitual_examined_all_pid,
                  diss_habitual, diss_habitual_examined_all, diss_habitual_not_examined_all, diss_hybrid, diss_mf,
                  diss_mb, diss_mcrl_examined_all, diss_mcrl_not_examined_all, not_examined_all_pid, examined_all_pid)
from scipy.stats import mannwhitneyu

def load_and_filter_data(file_path, participant_ids):
    """Load dataset from CSV file and filter by participant IDs."""
    df = pd.read_csv(file_path)
    return df[df['pid'].isin(participant_ids)]


def compute_average_score(df):
    """Compute average score per trial."""
    return df.groupby('trial_index').score.mean().reset_index()


def compute_last_10_avg_and_compare(data, analysis_type):
    """Compute the average score for the last 10 trials and compare groups using Mann-Whitney U test."""
    last_10_scores = {group: df[df['trial_index'] >= (df['trial_index'].max() - 59)]['score']
                      for group, df in data.items()}

    for group, scores in last_10_scores.items():
        print(f"Group {group}: Mean = {scores.mean():.2f}, Std = {scores.std():.2f}")

    comparisons = {}

    if analysis_type == "diss":
        comparison_pairs = [("habitual_not_all", "mcrl_not_all"),
                            ("habitual_examined_all", "mcrl_examined_all")]
    else:
        comparison_pairs = [("habitual_not_all", "habitual_examined_all"),
                            ("mf", "hybrid"),
                            ("habitual_not_all", "mf"),
                            ("habitual_examined_all", "hybrid")]

    for group1, group2 in comparison_pairs:
        if group1 in last_10_scores and group2 in last_10_scores:
            u_stat, p_value = mannwhitneyu(last_10_scores[group1], last_10_scores[group2], alternative='two-sided')
            comparisons[f'{group1} vs {group2}'] = {'U-statistic': u_stat, 'p-value': p_value}

    return comparisons

def plot_scores(data, analysis_type, save_path, ylim=(-180, 20)):
    """Plot the scores for different groups with labels and colors based on analysis type."""
    plt.figure(figsize=(7, 5))

    if analysis_type == "diss":
        labels_colors = {
            'examined_all': (
            f'Participant who examined \nall nodes', 'blue'),
            'not_examined_all': (
            f'Participant who did not \nexamine all nodes', 'red'),
        }
    else:
        labels_colors = {
            'habitual_not_all': (
            f'Habitual participant who did not examine all, n={len(habitual_not_examined_all_pid)}', 'red'),
            'habitual_examined_all': (
            f'Habitual participants who examined all, n={len(habitual_examined_all_pid)}', 'blue'),
            'mf': (f'Model-free participant, n={len(mf_pid)}', 'black'),
            'hybrid': (f'Hybrid participant, n={len(hybrid_pid)}', 'purple')
        }

    for group, df in data.items():
        label, color = labels_colors.get(group, ('Unknown', 'gray'))
        plt.plot(df['trial_index'], df['score'], label=label, color=color)

    plt.ylim(ylim)
    plt.xlabel('Trial', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def analyze(analysis_type, file_path, participant_groups, save_path):
    """Perform analysis based on the specified type."""
    filtered_data = {group: compute_average_score(load_and_filter_data(file_path, participant_groups[group]))
                     for group in participant_groups}

    plot_scores(filtered_data, analysis_type, save_path)
    comparisons = compute_last_10_avg_and_compare(filtered_data, analysis_type)
    print("Statistical comparisons (Mann-Whitney U test):", comparisons)



# Example usage
analysis = "diss"  # or "cogsci"
file_path = '../../data/human/strategy_discovery/mouselab-mdp.csv'

if analysis == "diss":
    participant_groups = {
        'habitual_not_all': diss_habitual_not_examined_all,
        'habitual_examined_all': diss_habitual_examined_all,
        'mcrl_not_all': diss_mcrl_not_examined_all,
        'mcrl_examined_all': diss_mcrl_examined_all,
    }
    save_path = 'plots/rldm_individual_difference_score.png'
else:
    participant_groups = {
        'habitual_not_all': habitual_not_examined_all_pid,
        'habitual_examined_all': habitual_examined_all_pid,
        'mf': mf_pid,
        'hybrid': hybrid_pid
    }
    save_path = 'individual_difference_score.png'

analyze(analysis, file_path, participant_groups, save_path)