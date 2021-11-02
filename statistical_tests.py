import sys
import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import pandas as pd
import numpy as np
import random
import pymannkendall as mk
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import friedmanchisquare, mannwhitneyu, ks_2samp, shapiro, bartlett, kruskal, ranksums
from mcl_toolbox.analyze_sequences import analyse_sequences
from mcl_toolbox.utils.statistics_utils import create_comparable_data

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

rpy2.robjects.numpy2ri.activate()
stats = importr("stats")

"""
This script runs statistical tests that tests whether:
1. strategy development and overall strategy frequency is significantly different across conditions
2. cluster development and overall cluster frequency is significantly different across conditions
3. decision system development and overall decision system frequency is significantly different across conditions

A mannwhitneyu / Wilcoxon rank sum test as well as Kolmogorov Smirnoff 2-sample test will be used to test whether the distributions of two independent samples are equal or not.
A friedmanchisquare test will be used to whether the distributions of two or more paired samples are equal or not.
A Mann Kendall test is used to test for trends
# todo: this file still needs restructuring: separate dataframe creation and the tests
"""


def create_data_for_distribution_test(strategy_name_dict: dict, block="training"):
    """
    Create data to check for equal distribution
    Args:
        strategy_name_dict: same as reward_exps:
        reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}

    Returns: dataframes containing all strategies and their count. Strategy number equal their corresponding description
    E.g. Strategy 21 has the number 21 in the dataframe and do not need to be added +1 anymore

    """
    # name: increasing
    # experiment v1.0
    column_names = ["increasing_variance", "decreasing_variance", "constant_variance"]
    cluster_df = pd.DataFrame(columns=column_names)
    strategy_df = pd.DataFrame(columns=column_names)
    decision_system_df = pd.DataFrame(columns=column_names)

    for (
        strategy_name,
        exp_num,
    ) in (
        strategy_name_dict.items()
    ):  # strategy_name: increasing/decreasing, exp_num: v1.0
        (
            strategy_proportions,
            _,
            cluster_proportions,
            _,
            decision_system_proportions,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = analyse_sequences(
            exp_num, number_of_trials=35, block=block, create_plot=False
        )
        strategy_df[strategy_name] = list(
            create_comparable_data(strategy_proportions, len=90).values()
        )
        cluster_df[strategy_name] = list(
            create_comparable_data(cluster_proportions, len=14).values()
        )
        decision_system_df[strategy_name] = decision_system_proportions[
            "Relative Influence (%)"
        ].tolist()
    return strategy_df, cluster_df, decision_system_df


def create_data_for_trend_test(
    reward_exps: dict, trend_test: True, number_of_strategies: int, block="training"
):
    """

    Args:
        reward_exps: a dictionary {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}
        trend_test: do you want to do a trend test?
        number_of_strategies: integer, 89
        block:

    Returns: trend data as pandas dataframes

    """

    column_names = list(reward_exps.keys())
    cluster_trend = pd.DataFrame(columns=column_names)
    strategy_trend = pd.DataFrame(columns=column_names)
    # decision_trend = pd.DataFrame(columns=column_names)
    adaptive_trend = pd.DataFrame(columns=column_names)
    maladaptive_trend = pd.DataFrame(columns=column_names)

    average_clicks = pd.DataFrame(columns=column_names)

    for strategy_name, exp_num in reward_exps.items():
        if exp_num == "c2.1":
            exp_num = "c2.1_dec"

        (
            strategy_proportions,
            strategy_proportions_trialwise,
            cluster_proportions,
            cluster_proportions_trialwise,
            decision_system_proportions,
            mean_dsw,
            top_n_strategies,
            worst_n_strategies,
            number_of_clicks,
            adaptive_participants,
            maladaptive_participants,
            other_participants,
            improved_participants,
        ) = analyse_sequences(
            exp_num,
            number_of_trials=35,
            block=block,
            create_plot=False,
            number_of_top_worst_strategies=number_of_strategies,
        )

        strategy_temp = []
        cluster_temp = []
        # ds_temp = []
        for i in range(0, len(strategy_proportions_trialwise)):
            strategy_temp.append(
                list(
                    create_comparable_data(
                        strategy_proportions_trialwise[i], len=90
                    ).values()
                )
            )
        if trend_test:
            strategy_trend[strategy_name] = list(
                map(list, zip(*strategy_temp))
            )  # transpose
        else:
            strategy_trend[strategy_name] = strategy_temp

        for i in range(0, len(cluster_proportions_trialwise)):
            cluster_temp.append(
                list(
                    create_comparable_data(
                        cluster_proportions_trialwise[i], len=14
                    ).values()
                )
            )
        if trend_test:
            cluster_trend[strategy_name] = list(map(list, zip(*cluster_temp)))
        else:
            cluster_trend[strategy_name] = cluster_temp

        # for i in range(0, len(mean_dsw)):
        #     ds_temp.append(list(create_comparable_data(mean_dsw[i], len=5).values()))
        # decision_trend[name] = ds_temp

        adaptive_trend[strategy_name] = top_n_strategies
        maladaptive_trend[strategy_name] = worst_n_strategies

        # Clicks
        number_of_clicks = number_of_clicks - 1
        average_clicks[exp_num] = number_of_clicks.mean(axis=1)
    return (
        strategy_trend,
        cluster_trend,
        adaptive_trend,
        maladaptive_trend,
        average_clicks,
    )


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
    print(f" ----------- Test for Equal distributions between the {type} -----------")

    if length_of_dict >= 3:
        stat, p = friedmanchisquare(*[v for k, v in name_distribution_dict.items()])
        print("Friedman chi-squared tests: stat=%.3f, p=%.3f" % (stat, p))

    for variance_type_a, distribution_a in name_distribution_dict.items():
        for variance_type_b, distribution_b in name_distribution_dict.items():
            stat, p = mannwhitneyu(distribution_a, distribution_b)
            print(
                f"Mann Whitney U: {variance_type_a} vs {variance_type_b}:  stat={stat:.3f}, p={p:.3f}"
            )

            stat, p = ks_2samp(distribution_a, distribution_b)
            print(
                f"Kolmogorov 2 sample : {variance_type_a} vs {variance_type_b}:  stat={stat:.3f}, p={p:.3f}"
            )


def test_for_trend(trend, analysis_type: str):
    # trend is a df with different conditions in each column
    # analysis_type: strategy or cluster or ds
    if trend.dtypes[0] == object:
        for columns in trend:  # increasing, decreasing, constant
            for strategy_number in range(0, trend.shape[0]):  # range(0, number of rows)
                test_results = mk.original_test(trend[columns][strategy_number])
                print(
                    f"Mann Kendall Test: {columns} {analysis_type}: ",
                    strategy_number,
                    test_results,
                )
    else:
        for columns in trend:
            test_results = mk.original_test(trend[columns])
            print(f"Mann Kendall Test: {columns} {analysis_type}: ", test_results)


def test_first_trials_vs_last_trials(trend, number_of_trials, analysis_type):
    """
    This function tests whether the distributions between the first n trials and the last n trials are equal.

    Args:
        trend: pandas dataframe with the variance types as header and number of trials as rows
        number_of_trials: number of first and last trials to take into consideration
        analysis_type: strategy or strategy cluster

    Returns: nothing but prints

    """

    # todo: add decision systems
    print(
        f" ----------------- Fisher Exact test: Do all {analysis_type} in the FIRST trials and all {analysis_type} "
        f"in the LAST trials have the same proportions WITHIN the same environment? -----------------"
    )
    average_first_n_trials = trend.iloc[0:number_of_trials].sum()  # add first n rows
    average_last_n_trials = trend.iloc[
        -(number_of_trials + 1) : -1, :
    ].sum()  # add last n rows
    for columns in trend:
        counts_first_n = (
            np.array(average_first_n_trials[columns])
            * number_of_participants
            * number_of_trials
        )
    counts_last_n = (
        np.array(average_last_n_trials[columns])
        * number_of_participants
        * number_of_trials
    )
    res = stats.fisher_test(
        np.array([counts_first_n, counts_last_n]), simulate_p_value=True
    )
    print(f"{columns} : p={res[0][0]:.3f}")


def test_last_n_across_environments(trend, number_of_trials, analysis_type):
    print(
        f" ----------------- Fisher Exact test: Do all {analysis_type} in the LAST N trials have the same proportion ACROSS environments? -----------------"
    )

    average_last_10_trials = trend.iloc[
        -(number_of_trials + 1) : -1, :
    ].sum()  # add last n rows
    for env_a in trend:
        for env_b in trend:
            counts_env_a = (
                np.array(average_last_10_trials[env_a])
                * number_of_participants
                * number_of_trials
            )
            counts_env_b = (
                np.array(average_last_10_trials[env_b])
                * number_of_participants
                * number_of_trials
            )
            res = stats.fisher_test(
                np.array([counts_env_a, counts_env_b]), simulate_p_value=True
            )
            print(f"{analysis_type} {env_a} vs {env_b}: p={res[0][0]:.3f}")


def test_of_proportions(
    env_distribution, analysis_type: str, individual_strategies=False
):
    """

    Args:
        env_distribution: if individual_strategies is false: a dictionary containing {env: array[proportions]}
        analysis_type:

    Returns:

    """
    if individual_strategies is False:
        print(
            f" ----------- Fisher exact test: Do {analysis_type} proportions of all {analysis_type} differ ACROSS environments?  ----------- "
        )
        for env_type_a, proportion_a in env_distribution.items():
            for env_type_b, proportion_b in env_distribution.items():
                # turn proportions into actual counts (times number of participants and number of trials)
                strategy_counts_a = (
                    np.array(proportion_a) * number_of_participants * number_of_trials
                )
                strategy_counts_b = (
                    np.array(proportion_b) * number_of_participants * number_of_trials
                )
                res = stats.fisher_test(
                    np.array([strategy_counts_a, strategy_counts_b]),
                    simulate_p_value=True,
                )
                print(
                    f"{analysis_type}: {env_type_a} vs {env_type_b}: p={res[0][0]:.3f}"
                )

                # stat, p = chisquare(distribution_a, distribution_b)
                # print(f"One sample Chi-squared tests : {variance_type_a} vs {variance_type_b}:  stat={stat:.3f}, p={p:.3f}")'

    else:
        print(
            f" ----------- Fisher exact test: Do {analysis_type} proportions of SINGLE {analysis_type} differ ACROSS environments?  ----------- "
        )
        for (
            index,
            strategies,
        ) in env_distribution.iterrows():  # loop through rows, i.e. strategies
            for (
                env_type_a
            ) in (
                env_distribution.columns
            ):  # loop through columns and get the column names
                for env_type_b in env_distribution.columns:
                    # skip if strategy vector is all 0
                    if (
                        np.count_nonzero(strategies[env_type_a]) < 2
                        or np.count_nonzero(strategies[env_type_b]) < 2
                    ):  # if 0 non-zero values are counted, i.e. all values are 0
                        print(
                            f"Strategy {index} has been skipped because of low frequency"
                        )
                        continue
                    else:
                        strategy_counts_a = (
                            np.array(strategies[env_type_a]) * number_of_participants
                            + 0.0
                        )
                        strategy_counts_b = (
                            np.array(strategies[env_type_b]) * number_of_participants
                            + 0.0
                        )
                        # print("COMBINED", np.array([strategy_counts_a, strategy_counts_b]))
                        res = stats.fisher_test(
                            np.array([strategy_counts_a, strategy_counts_b]),
                            simulate_p_value=True,
                        )
                        # print('p-value: {}'.format(res[0][0]))
                        print(
                            f"Strategy {index}: {env_type_a} vs {env_type_b}: p={res[0][0]:.3f}"
                        )

def anova_test(name_distribution_dict):
    # prepare the data
    # keys = name_distribution_dict.keys()
    df = pd.DataFrame.from_dict(name_distribution_dict)
    df_melt = pd.melt(df.reset_index(), id_vars=['index'])
    df_melt.columns = ['index', 'condition', 'clicks']
    #df_melt['clicks'] = ""
    #df_melt['reward_variance'] = ""


    # create new dfs for each condition
    for index, row in df_melt.iterrows():
        if row['condition'] == 'high_variance_high_cost':
            df_melt.at[index, 'click_cost'] = 'high'
            df_melt.at[index, 'reward_variance'] = 'high'
        if row['condition'] == 'high_variance_low_cost':
            df_melt.at[index, 'click_cost'] = 'low'
            df_melt.at[index, 'reward_variance'] = 'high'
        if row['condition'] == 'low_variance_low_cost':
            df_melt.at[index, 'click_cost'] = 'low'
            df_melt.at[index, 'reward_variance'] = 'low'
        if row['condition'] == 'low_variance_high_cost':
            df_melt.at[index, 'click_cost'] = 'high'
            df_melt.at[index, 'reward_variance'] = 'low'



    model = ols('clicks ~ C(click_cost) + C(reward_variance) + C(click_cost)*C(reward_variance)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # shapiro test to test for normal distribution of residuals; nullhypothesis: data is drawn from normal distribution
    w, pvalue = shapiro(model.resid)
    print(f"Shapiro test for normal distirbution of residuals: test-statistic: {w}, p-value: {pvalue}")

    # test for homogeneity of variances; nullhypothesis: samples from populations have equal variances
    w, pvalue = bartlett(df['high_variance_high_cost'], df['high_variance_low_cost'], df['low_variance_high_cost'], df['low_variance_low_cost'])
    print(f"Bartlett's test for normal distirbution of residuals: test-statistic: {w}, p-value: {pvalue}")

    return None

def equivalence_test(name_distribution_dict):
    for variance_type_a, distribution_a in name_distribution_dict.items():
        for variance_type_b, distribution_b in name_distribution_dict.items():
            p, v1, v2 = statsmodels.stats.weightstats.ttost_ind(distribution_a, distribution_b, -0.3, 0.3)
            print(f"Equivalence test : {variance_type_a} vs {variance_type_b}:  p={p:.3f}, lower test statistic={v1[0]:.3f}"
                  f", lower p-value={v1[1]:.3f}, upper test statistic={v2[0]:.3f}, lower p-value={v1[1]:.3f}")
    return None


if __name__ == "__main__":
    random.seed(123)
    number_of_trials = 35
    # number_of_participants = 14

    reward_exps = {"high_variance_low_cost": "high_variance_low_cost",
                   "high_variance_high_cost": "high_variance_high_cost",
                   "low_variance_low_cost": "low_variance_low_cost",
                   "low_variance_high_cost": "low_variance_high_cost"}  # cond 0  # cond 1
    # print(" --------------------------------------------------------------------")
    # print(" -------------------- Proportion Difference -------------------------")
    # print(" --------------------------------------------------------------------")
    # strategy_df, cluster_df, decision_system_df = create_data_for_distribution_test(reward_exps)
    #
    # print(
    #     f" ----------- This tests whether the proportions of all 89 strategies across environments are equal  -----------")
    # strategy_difference_dict = {"increasing": strategy_df["increasing_variance"],
    #                             "decreasing": strategy_df["decreasing_variance"],
    #                             "constant": strategy_df["constant_variance"]}
    # test_for_equal_distribution(strategy_difference_dict, "Strategies")
    # test_of_proportions(strategy_difference_dict, "Strategies", individual_strategies=False)
    #
    # print(
    #     f" ----------- This tests whether the proportions of all 13 strategy clusters across environments are equal  -----------")
    # cluster_difference_dict = {"increasing": cluster_df["increasing_variance"],
    #                            "decreasing": cluster_df["decreasing_variance"],
    #                            "constant": cluster_df["constant_variance"]}
    # test_for_equal_distribution(cluster_difference_dict, "Strategy Clusters")
    # test_of_proportions(cluster_difference_dict, "Strategies", individual_strategies=False)
    #
    # decision_system_difference_dict = {"increasing": decision_system_df["increasing_variance"],
    #                                    "decreasing": decision_system_df["decreasing_variance"],
    #                                    "constant": decision_system_df["constant_variance"]}
    # test_for_equal_distribution(decision_system_difference_dict, "Decision Systems")

    # print(" --------------------------------------------------------------------")
    # print(" ---------------------------- Trends --------------------------------")
    # print(" --------------------------------------------------------------------")
    (
        strategy_trend,
        cluster_trend,
        top_n_strategies,
        worst_n_strategies,
        number_of_clicks,
    ) = create_data_for_trend_test(
        reward_exps, number_of_strategies=5, trend_test=True
    )  # n adaptive, mal adaptive stratiges
    # test_for_trend(strategy_trend, "Strategy")
    # test_for_trend(cluster_trend, "Strategy Cluster")
    # test_for_trend(decision_trend, "Decision System")

    # print(" --------------------------------------------------------------------")
    # print(" ---------------------- First vs Last trial -------------------------")
    # print(" --------------------------------------------------------------------")
    # first_last_strategies, first_last_clusters, _, _, _ = create_data_for_trend_test(reward_exps, number_of_strategies=5,
    #                                                                               trend_test=False)
    # test_first_trials_vs_last_trials(first_last_strategies, 5, "Strategy")  # last 5 trials
    # test_first_trials_vs_last_trials(first_last_clusters, 5, "Strategy Cluster")
    #
    # test_last_n_across_environments(first_last_strategies, 5, "Strategy")
    # test_last_n_across_environments(first_last_strategies, 5, "Strategy Cluster")

    # print(
    #     " ----------------- Aggregated adaptive strategies vs. aggregated maladaptive strategies trends-----------------")
    # test_for_trend(top_n_strategies, "Adaptive Strategies")
    # test_for_trend(worst_n_strategies, "Maladaptive Strategies")
    #
    # test_of_proportions(top_n_strategies, "Adaptive Strategies", individual_strategies=False)
    # test_of_proportions(worst_n_strategies, "Maladaptive Strategies", individual_strategies=False)

    # print(
    #     " ----------------- strategies proportions-----------------")
    # test of proportions for only selected adaptive, maladaptive strategies
    # Do the proportions of the strategies differ across environment?
    # test_of_proportions(strategy_trend, "Strategies", individual_strategies=True)

    print(" -----------------Number of clicks-----------------")

    # test_for_trend(number_of_clicks, "Clicks")

    ### ANOVA and t-test require normality and equal variance. therefore now Kruskal-Willis test and Wilcoxon rank sum test
    # print("Kruskal Wallis test", kruskal(number_of_clicks["high_variance_high_cost"].tail(5), number_of_clicks["high_variance_low_cost"].tail(5),
    #               number_of_clicks["low_variance_high_cost"].tail(5), number_of_clicks["high_variance_low_cost"].tail(5)))
    # print("Wilcoxon ranksum - High variance ", ranksums(number_of_clicks["high_variance_high_cost"], number_of_clicks["high_variance_low_cost"]))
    # print("Wilcoxon ranksum - Low variance ", ranksums(number_of_clicks["low_variance_high_cost"], number_of_clicks["low_variance_low_cost"]))
    # print("Wilcoxon ranksum - High cost ", ranksums(number_of_clicks["high_variance_high_cost"], number_of_clicks["low_variance_high_cost"]))
    # print("Wilcoxon ranksum - Low cost ", ranksums(number_of_clicks["high_variance_low_cost"], number_of_clicks["low_variance_low_cost"]))

# print("# of clicks at the beginning of the trial vs. # of clicks at the end of the trial for both cond")
    # statistical tests: # of clicks at the beginning of the trial vs. # of clicks at the end of the trial for both cond
    # print(
    #     "High variance - Low cost condition - first 5 vs last 5",
    #     ranksums(
    #         number_of_clicks["high_variance_low_cost"].head(5),
    #         number_of_clicks["high_variance_low_cost"].tail(5)
    #     ),
    # )
    # print(
    #     "High variance - High cost condition - first 5 vs last 5",
    #     ranksums(
    #         number_of_clicks["high_variance_high_cost"].head(5),
    #         number_of_clicks["high_variance_high_cost"].tail(5)
    #     ),
    # )
    # print(
    #     "Low variance - High cost condition - first 5 vs last 5",
    #     ranksums(
    #         number_of_clicks["low_variance_high_cost"].head(5),
    #         number_of_clicks["low_variance_high_cost"].tail(5)
    #     ),
    # )
    # print(
    #     "Low variance - Low cost condition - first 5 vs last 5",
    #     ranksums(
    #         number_of_clicks["low_variance_low_cost"].head(5),
    #         number_of_clicks["low_variance_low_cost"].tail(5)
    #     ),
    # )

    # print("# of clicks at the end of the trial in cond 0 vs cond 1")
    # statistical tests: # of clicks at the end of the trial in cond 0 vs cond 1
    # print(
    #     "High variance - last 5 trials",
    #     ranksums(
    #         number_of_clicks["high_variance_high_cost"],
    #         number_of_clicks["high_variance_low_cost"]
    #     ),
    # )
    # print(
    #     "Low variance - last 5 trials",
    #     ranksums(
    #         number_of_clicks["low_variance_high_cost"],
    #         number_of_clicks["low_variance_low_cost"]
    #     ),
    # )
    # print(
    #     "High cost - last 5 trials",
    #     ranksums(
    #         number_of_clicks["high_variance_high_cost"],
    #         number_of_clicks["low_variance_high_cost"]
    #     ),
    # )
    # print(
    #     "Low cost - last 5 trials",
    #     ranksums(
    #         number_of_clicks["high_variance_low_cost"],
    #         number_of_clicks["low_variance_low_cost"]
    #     ),
    # )

    #### Equivalence test ####
    # equivalence_test(number_of_clicks)