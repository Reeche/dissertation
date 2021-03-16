import sys
import pandas as pd
import numpy as np
import random
import pymannkendall as mk
from scipy.stats import friedmanchisquare
from scipy.stats import mannwhitneyu

from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

from mcl_toolbox.utils.statistics_utils import create_data_for_distribution_test, create_data_for_trend_test

"""
This script runs statistical tests that tests whether:
1. strategy development and overall strategy frequency is significantly different across conditions
2. cluster development and overall cluster frequency is significantly different across conditions
3. decision system development and overall decision system frequency is significantly different across conditions

A mannwhitneyu test will be used to test whether the distributions of two independent samples are equal or not.
A friedmanchisquare test will be used to whether the distributions of two or more paired samples are equal or not.
A Mann Kendall test is used to test for trends

"""


# todo: test for equal distribution need to be changed to chi squared test of independence
def test_for_equal_distribution(name_distribution_dict: dict, type: str):
    """
    Are the distributions of the proportions between the environments equal?
    friedmanchisquare for all and mannwhitneyu tests for pairs

    Args:
        name_distribution_dict: a dictionary with the variance type and corresponding distribution }
        e.g. {"increasing": [array]}

    Returns: None

    """
    length_of_dict = len(name_distribution_dict)
    print(f" === Test for Equal distributions between the {type} === ")

    if length_of_dict >= 3:
        stat, p = friedmanchisquare(*[v for k, v in name_distribution_dict.items()])
        print('Friedman chi-squared tests: stat=%.3f, p=%.3f' % (stat, p))

    for variance_type_a, distribution_a in name_distribution_dict.items():
        for variance_type_b, distribution_b in name_distribution_dict.items():
            stat, p = mannwhitneyu(distribution_a, distribution_b)
            print(f"{variance_type_a} vs {variance_type_b}:  stat={stat:.3f}, p={p:.3f}")

def test_for_trend(trend, analysis_type: str):
    # analysis_type: strategy or cluster or ds
    for columns in trend:
        for strategy_number in range(0, len(trend["increasing_variance"])):
            test_results = mk.original_test(trend[columns][strategy_number])
            print(f"Mann Kendall Test: {columns} {analysis_type}: ", strategy_number, test_results)

def test_first_trials_vs_last_trials(trend, n, analysis_type):
    """
    This function tests whether the distributions between the first n trials and the last n trials are equal.
    Args:
        trend: pandas dataframe with the variance types as header and number of trials as rows
        n: number of first and last trials to take into consideration
        analysis_type: strategy or strategy cluster

    Returns:

    """
    # todo: add decision systems
    average_first_n_trials = trend.iloc[0:n].sum()  # add first n rows
    average_last_n_trials = trend.iloc[-(n + 1):-1, :].sum()  # add last n rows
    for columns in trend:
        stat, p = mannwhitneyu(average_first_n_trials[columns], average_last_n_trials[columns])
        print(f'{analysis_type}, {columns}: First{n} trials vs Last {n} trials: stat=%.3f, p=%.3f' % (stat, p))



if __name__ == "__main__":
    random.seed(123)
    reward_exps = {"increasing_variance": "v1.0",
                   "decreasing_variance": "c2.1_dec",
                   "constant_variance": "c1.1"}

    print(" ----------------- Distribution Difference -----------------")
    strategy_df, cluster_df, decision_system_df = create_data_for_distribution_test(reward_exps)

    strategy_difference_dict = {"increasing": strategy_df["increasing_variance"],
                                "decreasing": strategy_df["decreasing_variance"],
                                "constant": strategy_df["constant_variance"]}
    test_for_equal_distribution(strategy_difference_dict, "Strategies")

    cluster_difference_dict = {"increasing": cluster_df["increasing_variance"],
                               "decreasing": cluster_df["decreasing_variance"],
                               "constant": cluster_df["constant_variance"]}
    test_for_equal_distribution(cluster_difference_dict, "Strategy Clusters")
    #
    # decision_system_difference_dict = {"increasing": decision_system_df["increasing_variance"],
    #                                    "decreasing": decision_system_df["decreasing_variance"],
    #                                    "constant": decision_system_df["constant_variance"]}
    # test_for_equal_distribution(decision_system_difference_dict, "Decision Systems")

    print(" ----------------- Trends -----------------")
    strategy_trend, cluster_trend = create_data_for_trend_test(reward_exps, trend_test=True)
    test_for_trend(strategy_trend, "Strategy")
    test_for_trend(cluster_trend, "Strategy Cluster")
    #test_for_trend(decision_trend, "Decision System")

    # print(" ----------------- Decision System -----------------")
    # for i in range(0, 5):
    #     increasing_ds_trend = mk.original_test(list(mean_dsw_increasing[:, i]))
    #     print("Mann Kendall Test: Increasing Decision System: ", i, increasing_ds_trend)
    #
    # for i in range(0, 5):
    #     decreasing_ds_trend = mk.original_test(list(mean_dsw_decreasing[:, i]))
    #     print("Mann Kendall Test: Decreasing Decision System: ", i, decreasing_ds_trend)
    #
    # for i in range(0, 5):
    #     constant_ds_trend = mk.original_test(list(mean_dsw_constant[:, i]))
    #     print("Mann Kendall Test: Constant Decision System: ", i, constant_ds_trend)

    print(" ----------------- First vs Last trial -----------------")
    first_last_strategies, first_last_clusters = create_data_for_trend_test(reward_exps, trend_test=False)
    test_first_trials_vs_last_trials(first_last_strategies, 2, "Strategy")
    test_first_trials_vs_last_trials(first_last_clusters, 2, "Strategy Cluster")
    #get_first_n_trials(decision_trend, 2, "Decision System")

    # print(
    #     " ----------------- Aggregated adaptive strategies vs. aggregated maladaptive strategies trends-----------------")
    # filtered_strategies = filter_used_strategies(reward_exps)
    # top_n, worst_n = adaptive_maladaptive_filtered_strategies(reward_exps, filtered_strategies, 5)


    # when it comes to plotting, the strategy names need to equal the description
    # # todo: make this as a function
    # for strategy_name, exp_num in reward_exps.items():
    #     top_n_increasing = top_n[strategy_name]
    #     adaptive_strategy_list = list(top_n_increasing.keys())
    #
    #     worst_n_increasing = worst_n[strategy_name]
    #     maladaptive_strategy_list = list(worst_n_increasing.keys())
    #
    #     # filter strategy_df so that it only contains the adaptive/maladaptive strategies and sum them up
    #     #print(strategy_df[strategy_name].loc[adaptive_strategy_list])
    #
    #
    #     _, _, _, _, _, _, adaptive_strategy_sum, maladaptive_strategy_sum, rest_strategy_sum = load_data_from_computational_microscope(strategy_name, exp_num, reward_exps)
    #
    #     strategy_difference_dict = {strategy_name: adaptive_strategy_sum}
    #     test_for_equal_distribution(strategy_difference_dict, "Strategies")
