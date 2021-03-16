import pandas as pd
#from mcl_toolbox.utils.experiment_utils import analyse_sequences

def replace_none_with_empty_str(some_dict):
    return {k: (0 if v is None else v) for k, v in some_dict.items()}

# create empty dictionary so the clusters can be compared
def create_comparable_data(proportions, len):
    """
    This function creates clusters/decision system of equal length so that they can be compared.
    E.g. cluster_a = {1: 0.1, 3:0.5, 12:0.2}
    None values will be replaced by 0.

    Args:
        proportions: which porportions, e.g. cluster or strategy
        len: number of the cluster (13) or strategy (89)

    Returns: dict of certain length. For the example: {1: 0.1, 2: 0,..., 12:0.3, 13:0}

    """
    _keys = list(range(0, len))
    _dict = {key: None for key in _keys}

    for keys, values in _dict.items():
        value_new = proportions.get(keys, None)
        if value_new:
            _dict[keys] = value_new

    _dict = replace_none_with_empty_str(_dict)
    return _dict

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

    for strategy_name, exp_num in strategy_name_dict.items():  # strategy_name: increasing/decreasing, exp_num: v1.0
        strategy_proportions, _, cluster_proportions, _, decision_system_proportions, _ = analyse_sequences(exp_num, block=block, create_plot=False)
        strategy_df[strategy_name] = list(create_comparable_data(strategy_proportions, len=89).values())
        cluster_df[strategy_name] = list(create_comparable_data(cluster_proportions, len=14).values())
        decision_system_df[strategy_name] = decision_system_proportions["Relative Influence (%)"].tolist()
    return strategy_df, cluster_df, decision_system_df


def create_data_for_trend_test(reward_exps: dict, trend_test: True, block="training"):
    """
    Create data for the trend tests
    Args:
        strategy_name_dict: reward_exps
        reward_exps = {"increasing_variance": "v1.0",
               "decreasing_variance": "c2.1",
               "constant_variance": "c1.1"}

    Returns: trend data as pandas dataframes
    #todo: decision trend, not really used but might be useful for completeness
    """
    column_names = ["increasing_variance", "decreasing_variance", "constant_variance"]
    cluster_trend = pd.DataFrame(columns=column_names)
    strategy_trend = pd.DataFrame(columns=column_names)
    # decision_trend = pd.DataFrame(columns=column_names)

    for strategy_name, exp_num in reward_exps.items():
        _, strategy_proportions_trialwise, _, cluster_proportions_trialwise, _, mean_dsw = analyse_sequences(exp_num, block=block, create_plot=False)

        strategy_temp = []
        cluster_temp = []
        ds_temp = []
        for i in range(0, len(strategy_proportions_trialwise)):
            strategy_temp.append(list(create_comparable_data(strategy_proportions_trialwise[i], len=89).values()))
        if trend_test:
            strategy_trend[strategy_name] = list(map(list, zip(*strategy_temp)))  # transpose
        else:
            strategy_trend[strategy_name] = strategy_temp

        for i in range(0, len(cluster_proportions_trialwise)):
            cluster_temp.append(list(create_comparable_data(cluster_proportions_trialwise[i], len=14).values()))
        if trend_test:
            cluster_trend[strategy_name] = list(map(list, zip(*cluster_temp)))
        else:
            cluster_trend[strategy_name] = cluster_temp

        # for i in range(0, len(mean_dsw)):
        #     ds_temp.append(list(create_comparable_data(mean_dsw[i], len=5).values()))
        # decision_trend[name] = ds_temp

    return strategy_trend, cluster_trend  # , decision_trend




