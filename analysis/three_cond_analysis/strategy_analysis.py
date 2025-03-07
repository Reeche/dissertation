import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import statsmodels.api as sm
from vars import clicking_pid, assign_model_names
import pymannkendall as mk
import ast
import numpy as np


def plot_all_strategy_proportions(data, model, mapping: dict):
    df = data.replace(mapping)

    ### new code to verify that the old code is correct, but the coloring in the new one is a bit off, so stick to the old plotting
    # # for each column, count how often each unique value appears as count
    # value_counts = df.apply(lambda x: x.value_counts()).fillna(0).astype(int)
    #
    # # reshape so df has the columns "trial", "adaptive", "mod", "mal"
    # value_counts = value_counts.transpose()
    #
    # # for each column, get the proportion as well as 95% CI as proportion of the column
    # for column in value_counts:
    #     value_counts[column] = value_counts[column] / len(df)
    #     std_dev = np.std(value_counts[column], axis=0)
    #     n = len(value_counts[column])
    #     std_err = std_dev / np.sqrt(n)
    #     conf_interval = 1.96 * std_err
    #     value_counts[column + "_CI"] = conf_interval
    #
    # # plot the proportions
    # plt.figure(figsize=(8, 6))
    # plt.plot(value_counts[["adaptive", "mod", "mal"]], label=["Adaptive", "Mod. adaptive", "Maladaptive"])
    # # plot the confidence intervals
    # for column in value_counts:
    #     if "CI" in column:
    #         plt.fill_between(value_counts.index, value_counts[column.replace("_CI", "")] - value_counts[column],
    #                          value_counts[column.replace("_CI", "")] + value_counts[column], alpha=0.2)
    # plt.ylim(-0.1, 1)
    # plt.xlabel("Trials", fontsize=14)
    # plt.ylabel("Proportion", fontsize=14)
    # plt.title(f"Strategy proportions for {model}")
    # plt.legend(fontsize=14)
    # # plt.savefig(f"plots/CM_all_strategies/{model}_cm.png")
    # plt.show()
    # plt.close()
    # return None

    # figure size
    plt.figure(figsize=(8, 6))
    frequencies = pd.DataFrame(columns=["Adaptive", "Mod. adaptive", "Maladaptive"])
    for columns in df:
        frequencies = frequencies._append({'Adaptive': Counter(df[columns])["adaptive"] / len(df),
                                           'Mod. adaptive': Counter(df[columns])["mod"] / len(df),
                                           'Maladaptive': Counter(df[columns])["mal"] / len(df)}, ignore_index=True)

    # trend_test(frequencies)

    # translate proportions to percentages
    frequencies = frequencies * 100

    # add the frequency of each strategy to the plot
    plt.plot(frequencies, label=[
        f"Adaptive, {frequencies['Adaptive'].iloc[0].round(2)}% to {frequencies['Adaptive'].iloc[-1].round(2)}%",
        f"Mod. adaptive, {frequencies['Mod. adaptive'].iloc[1].round(2)}% to {frequencies['Mod. adaptive'].iloc[-1].round(2)}%",
        f"Maladaptive, {frequencies['Maladaptive'].iloc[2].round(2)}% to {frequencies['Maladaptive'].iloc[-1].round(2)}%"])

    # get the std error for each column
    std_err = frequencies.std() / (len(frequencies) ** 0.5)
    error_margin = 1.96 * std_err
    # Confidence interval for each column
    for column in frequencies:
        plt.fill_between(frequencies.index, frequencies[column] - error_margin[column],
                         frequencies[column] + error_margin[column], alpha=0.2)
    plt.ylim(-10, 100)
    plt.xlabel("Trials", fontsize=14)
    plt.ylabel("Proportion", fontsize=14)
    # plt.title(f"Strategy proportions for {model}")
    plt.legend(fontsize=14)
    plt.savefig(f"plots/CM_pid/{experiment}_{model}_cm.png")
    plt.show()
    plt.close()
    return None


def compare_last_trial_proportions(participants: dict):
    ## get last row of strategies
    last_strategy = []
    for key, value in participants.items():
        if isinstance(value, list) and len(value) > 0:
            last_strategy.append(value[-1])
    # Use map() function to replace each item from the list
    mapped_list = list(map(mapping_dict.get, last_strategy))
    counter = Counter(mapped_list)
    print("Participants")
    for key in counter.keys():
        print(key, counter[key] / len(mapped_list))


def clustering_kmeans(strategy_scores):
    kmeans = KMeans(n_clusters=3, n_init="auto", max_iter=10000, random_state=42)
    kmeans.fit(strategy_scores.values)
    strategy_scores["label"] = kmeans.labels_

    ## relabel the cluster centers
    cluster_centers = pd.Series(kmeans.cluster_centers_.flatten())
    cluster_centers = cluster_centers.sort_values()
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[0]), "mal")
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[1]), "mod")
    strategy_scores["label"] = strategy_scores["label"].replace(int(cluster_centers.index[2]), "adaptive")
    return strategy_scores


def classify_strategies(strategies, experiment):
    # clustering of all available strategies, not only used ones

    if experiment == "v1.0":
        mapping_dict = {
            "adaptive": [21, 63, 88, 40, 51, 50, 38, 16, 83, 80, 29, 67, 47, 64, 58, 43, 24, 49, 42, 25, 57, 41, 31, 37,
                         75, 76, 19, 68, 85, 18, 17, 20, 87, 84, 52, 48, 62, 60, 77, 82, 59, 3, 9, 8, 10, 15, 5, 4, 54,
                         2, 36, 1, 71, 7, 11, 14, 46, 6, 72, 12, 13, 65, 26, 78, 45, 55, 56, 35, 86],
            "mal": [33, 73, 32, 79, 89, 69, 44, 34, 81, 27, 61],
            "mod": [74, 66, 22, 28, 70, 53, 30, 23, 39]}
    elif experiment == "c2.1":
        mapping_dict = {
            "adaptive": [70, 23, 69, 65, 32, 33, 81, 37, 25, 79, 53, 22, 34, 31, 47, 64, 49, 80, 63, 48, 62, 84, 13, 54,
                         10, 11, 14, 71, 3, 82, 36, 1, 5, 2, 7, 15, 6, 72, 12, 8, 9, 4, 46, 45, 60],
            "mal": [74, 66, 78, 21, 86, 26, 89, 27, 73, 52, 77, 30, 56, 55, 67, 58, 88, 87, 85, 41, 57, 16, 29, 38, 76,
                    50, 24, 40, 51, 43, 42, 83, 39],
            "mod": [75, 68, 19, 20, 18, 17, 28, 61, 35, 44, 59]}
    elif experiment == "c1.1":
        mapping_dict = {
            "adaptive": [65, 33, 81, 34, 21, 69, 64, 63, 25, 32, 88, 79, 16, 37, 29, 86, 26, 49, 83, 80, 51, 38, 50, 35,
                         52, 77, 78, 40, 84, 47, 57, 59, 82, 45],
            "mod": [19, 17, 18, 20, 68, 67, 75, 55, 56, 70, 22, 43, 58, 76, 53, 87, 62, 85, 48, 73, 61, 44, 60, 41, 31,
                    23, 89, 74, 7, 6, 11, 12, 72, 10, 14, 36, 9, 71, 2, 1, 13, 3, 5, 8, 15, 46, 4, 54, 42, 24, 27, 28,
                    66, 30],
            "mal": [39]}

    # Iterate over each key-value pair in the mapping dictionary
    for replacement, values in mapping_dict.items():
        # Replace values in the DataFrame for each key-value pair
        strategies.replace({col: {val: replacement for val in values} for col in strategies.columns}, inplace=True)

    return strategies


def frequency(kmeans_results):
    counter = Counter(kmeans_results.labels_)
    print("Simulation")
    for key in counter.keys():
        print(key, counter[key] / len(kmeans_results.labels_))


def adaptive_proportion_higher_than_chance(strategy_labels, participants_df):
    ## count how many
    cm_counts = (strategy_labels["label"].value_counts() / sum(strategy_labels["label"].value_counts())) * 100
    participants_counter = participants_df.iloc[:, -1:].replace(mapping_dict)
    participant_counts = (participants_counter.value_counts() / sum(participants_counter.value_counts())) * 100
    non_adaptive_cm = cm_counts["mal"] + cm_counts["mod"]
    if "mal" in participant_counts.index and "mod" in participant_counts.index:
        non_adaptive_pid = participant_counts["mal"] + participant_counts["mod"]
    elif "mal" in participant_counts.index and "mod" not in participant_counts.index:
        non_adaptive_pid = participant_counts["mal"]
    elif "mod" in participant_counts.index and "mal" not in participant_counts.index:
        non_adaptive_pid = participant_counts["mod"]
    print("pid", participant_counts["adaptive"], non_adaptive_pid)
    print("CM", cm_counts["adaptive"], non_adaptive_cm)
    res = chisquare([participant_counts["adaptive"], non_adaptive_pid], f_exp=[cm_counts["adaptive"], non_adaptive_cm])
    print(res)


def logistic_regression(df, mapping_dict):
    # replace the column strategy with the mapping selected from the column condition
    # df["strategy"] = df.apply(lambda x: mapping_dict[x["strategy"]], axis=1)

    # replace strategy with 1 for adaptive and 0 for anything else
    df["strategy"] = [1 if x == "adaptive" else 0 for x in df["strategy"]]

    # logistic regression
    # model = sm.GLM.from_formula("strategy ~ C(condition, Treatment('c1.1'))*trial", data=participants_df,
    #                             family=sm.families.Binomial()).fit()
    # print(model.summary())

    ### logisit regression for each condition
    for condition in pd.unique(df["condition"]):
        condition_df = df[df["condition"] == condition]
        model = sm.GLM.from_formula("strategy ~ trial", data=condition_df,
                                    family=sm.families.Binomial()).fit()
        print(condition)
        print(model.summary())

    return None


def trend_test(data):
    result = mk.original_test(data["Adaptive"])
    print(result)
    return None


# def plot_adaptive_proportion(data, experiment, pid_mapping):
#     data['model'] = data.apply(assign_model_names, axis=1)
#     data = data[["model", "model_strategies", "pid_strategies"]]
#
#     data['model_strategies'] = data['model_strategies'].apply(ast.literal_eval)
#     data['pid_strategies'] = data['pid_strategies'].apply(ast.literal_eval)
#
#     # plt.figure(figsize=(8, 6))
#     x = np.arange(0, 35)
#
#     for model_name in ["Non-learning", "SSL", "Habitual"]:
#         data_filtered = data[data["model"] == model_name]
#         data_temp = pd.DataFrame(data_filtered['model_strategies'].tolist(), columns=[f'{i + 1}' for i in range(35)])
#         ##clustering based on clusters used for participants
#         data_temp = classify_strategies(data_temp, experiment)
#
#         # for each column count how often "adaptive" appears divided by total length of the column
#         adaptive_proportion = data_temp.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#         plt.plot(x, adaptive_proportion, label=model_name)
#
#         result = mk.original_test(adaptive_proportion)
#         # print(f"{model_name}:", adaptive_proportion)
#         print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")
#
#     ### PID data
#     adaptive_proportion_pid = pid_mapping.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#
#     # Calculate mean and standard error for each data point
#     std_dev = np.std(adaptive_proportion_pid, axis=0)
#     n = len(adaptive_proportion_pid)
#     std_err = std_dev / np.sqrt(n)
#
#     # Calculate the confidence interval
#     conf_interval = 1.96 * std_err
#
#     x = np.arange(0, len(adaptive_proportion_pid))
#
#     plt.plot(adaptive_proportion_pid, label="Participant", color="blue", linewidth=3)
#     plt.fill_between(x, adaptive_proportion_pid - conf_interval, adaptive_proportion_pid + conf_interval, color='blue',
#                      alpha=0.1,
#                      label='95% CI')
#
#     plt.xlabel("Trial", fontsize=12)
#     plt.ylim(0, 1)
#     plt.ylabel("Proportion of adaptive strategies", fontsize=12)
#     plt.legend(fontsize=11, ncol=2)
#     # plt.savefig(f"plots/{experiment}/alternatives_proportions.png")
#     plt.show()
#     plt.close()
#
#     for model_name in ["hybrid LVOC", "hybrid Reinforce", "MF - LVOC", "MF - Reinforce"]:
#         data_filtered = data[data["model"] == model_name]
#         data_temp = pd.DataFrame(data_filtered['model_strategies'].tolist(), columns=[f'{i + 1}' for i in range(35)])
#         ##clustering based on clusters used for participants
#         data_temp = classify_strategies(data_temp, experiment)
#
#         # for each column count how often "adaptive" appears divided by total length of the column
#         adaptive_proportion = data_temp.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#         plt.plot(x, adaptive_proportion, label=model_name)
#
#         result = mk.original_test(adaptive_proportion)
#         print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")
#
#     ### PID data
#     adaptive_proportion_pid = pid_mapping.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#
#     # Calculate mean and standard error for each data point
#     std_dev = np.std(adaptive_proportion_pid, axis=0)
#     n = len(adaptive_proportion_pid)
#     std_err = std_dev / np.sqrt(n)
#
#     # Calculate the confidence interval
#     conf_interval = 1.96 * std_err
#
#     x = np.arange(0, len(adaptive_proportion_pid))
#
#     plt.plot(adaptive_proportion_pid, label="Participant", color="blue", linewidth=3)
#     plt.fill_between(x, adaptive_proportion_pid - conf_interval, adaptive_proportion_pid + conf_interval, color='blue',
#                      alpha=0.1,
#                      label='95% CI')
#
#     plt.xlabel("Trial", fontsize=12)
#     plt.ylim(0, 1)
#     plt.ylabel("Proportion of adaptive strategies", fontsize=12)
#     plt.legend(fontsize=11, ncol=2)
#     # plt.savefig(f"plots/{experiment}/MF_proportions.png")
#     plt.show()
#     plt.close()
#
#     for model_name in ["MB - No assump., grouped", "MB - No assump., ind.",
#                        "MB - Uniform, ind.", "MB - Uniform, grouped",
#                        "MB - Level, grouped", "MB - Level, ind."]:
#         data_filtered = data[data["model"] == model_name]
#         data_temp = pd.DataFrame(data_filtered['model_strategies'].tolist(), columns=[f'{i + 1}' for i in range(35)])
#         ##clustering based on clusters used for participants
#         data_temp = classify_strategies(data_temp, experiment)
#
#         # for each column count how often "adaptive" appears divided by total length of the column
#         adaptive_proportion = data_temp.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#         plt.plot(x, adaptive_proportion, label=model_name)
#
#         result = mk.original_test(adaptive_proportion)
#         print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")
#
#         ### PID data
#
#     adaptive_proportion_pid = pid_mapping.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)
#
#     # Calculate mean and standard error for each data point
#     std_dev = np.std(adaptive_proportion_pid, axis=0)
#     n = len(adaptive_proportion_pid)
#     std_err = std_dev / np.sqrt(n)
#
#     # Calculate the confidence interval
#     conf_interval = 1.96 * std_err
#
#     x = np.arange(0, len(adaptive_proportion_pid))
#
#     plt.plot(adaptive_proportion_pid, label="Participant", color="blue", linewidth=3)
#     plt.fill_between(x, adaptive_proportion_pid - conf_interval, adaptive_proportion_pid + conf_interval, color='blue',
#                      alpha=0.1,
#                      label='95% CI')
#
#     plt.xlabel("Trial", fontsize=12)
#     plt.ylim(0, 1)
#     plt.ylabel("Proportion of adaptive strategies", fontsize=12)
#     plt.legend(fontsize=11, ncol=2)
#     # plt.savefig(f"plots/{experiment}/MB_proportions.png")
#     plt.show()
#     plt.close()

def calculate_and_plot(data, model_names, experiment, pid_mapping):
    """Calculate adaptive proportions and plot for the given model names."""
    x = np.arange(0, 35)
    for model_name in model_names:
        data_filtered = data[data["model"] == model_name]
        data_temp = pd.DataFrame(data_filtered['model_strategies'].tolist(), columns=[f'{i + 1}' for i in range(35)])
        data_temp = classify_strategies(data_temp, experiment)

        adaptive_proportion = data_temp.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)

        ## trend test
        # result = mk.original_test(adaptive_proportion)
        # print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # add proportion of the first and last trial as percentage rounded to 2 decimals to the label
        first_trial = round(adaptive_proportion.iloc[0] * 100, 2)
        last_trial = round(adaptive_proportion.iloc[-1] * 100, 2)
        label = f"{model_name}, {first_trial}% to {last_trial}%"
        # label = f"{model_name}"
        plt.plot(x, adaptive_proportion, label=label)

    adaptive_proportion_pid = pid_mapping.apply(lambda x: x.value_counts(normalize=True).get("adaptive", 0), axis=0)

    std_dev = np.std(adaptive_proportion_pid, axis=0)
    n = len(adaptive_proportion_pid)
    std_err = std_dev / np.sqrt(n)
    conf_interval = 1.96 * std_err

    plt.plot(adaptive_proportion_pid, label="Participant", color="blue", linewidth=3)
    plt.fill_between(np.arange(0, len(adaptive_proportion_pid)),
                     adaptive_proportion_pid - conf_interval,
                     adaptive_proportion_pid + conf_interval,
                     color='blue', alpha=0.1, label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    plt.ylim(0, 1)
    plt.ylabel("Proportion of adaptive strategies", fontsize=12)
    plt.legend(fontsize=10, ncol=2)
    plt.savefig(f"plots/CM/{experiment}_{model_names}_adaptive_proportions.png")
    # plt.show()
    plt.close()


def plot_adaptive_proportion(data, experiment, pid_mapping):
    """Main function to plot adaptive proportions for all models."""
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

    for group, models in model_groups.items():
        calculate_and_plot(data, models, experiment, pid_mapping)


if __name__ == "__main__":
    # experiments = ["v1.0", "c2.1", "c1.1"]
    experiments = ["v1.0"]
    all_pid = pd.DataFrame()
    mapping_dict = {}
    for experiment in experiments:
        strategy_scores = pd.read_pickle(f"../../results/strategy_scores/{experiment}_strategy_scores.pkl")
        strategy_scores = pd.DataFrame.from_dict(strategy_scores, orient='index')
        strategy_scores.index += 1  # increment by one because participants start at 1

        # ### load participanta data
        participants = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")
        # # filter for clicking participants
        participants = {key: value for key, value in participants.items() if key in clicking_pid[experiment]}
        participants_df = pd.DataFrame.from_dict(participants, orient='index')

        # ## get only used strategies
        unique_used_strategies = pd.unique(participants_df.values.flatten())

        # ##plot strategy proportions for participants
        # # k means based on used strategy scores
        used_strategy_score = strategy_scores.loc[unique_used_strategies]  # important to use loc here
        strategy_labels = clustering_kmeans(used_strategy_score)
        mapping = used_strategy_score.set_index(strategy_labels.index)['label']

        # ### clustering by using all strategies
        pid_mapping = classify_strategies(participants_df, experiment)
        # plot_all_strategy_proportions(participants_df, "pid", mapping)

        #  ## load CM model data
        model_data = pd.read_csv(f"../../final_results/model_cm_300_fit/{experiment}.csv")
        # # filter for clicking participants
        model_data = model_data[model_data["pid"].isin(clicking_pid[experiment])]
        #
        # # adaptive_proportion_higher_than_chance(strategy_labels, participants_df)
        plot_adaptive_proportion(model_data, experiment, pid_mapping)

        ### reshape df for logisitic regression
        # df = participants_df.transpose()
        # df = pd.melt(df)
        # df.columns = ["pid", "strategy"]
        # df["trial"] = df.groupby("pid").cumcount() + 1
        # df["condition"] = experiment
        #
        # logistic_regression(df, mapping)
        #
        # all_pid = all_pid._append(df)
        # mapping_dict[experiment] = mapping

    # logistic_regression(all_pid, mapping_dict)
    #
