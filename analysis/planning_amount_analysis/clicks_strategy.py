from number_of_click_analysis import create_click_df
import pandas as pd
import pymannkendall as mk
import statsmodels.formula.api as smf


def get_change_indices(numbers):
    change_indices = []

    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1]:
            change_indices.append(i)

    return change_indices


def load_click_data(experiment):
    data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
    complete_click_df = create_click_df(data, experiment)
    click_df = complete_click_df[['pid', 'trial', 'number_of_clicks']].copy()
    reshaped_click_df = click_df.pivot(index="trial", columns="pid", values="number_of_clicks")
    reshaped_click_df.columns = reshaped_click_df.columns.map(str)
    return reshaped_click_df


def load_strategy_type_data(experiment):
    cluster_name_mapping = {1: 1,
                            2: 2,
                            3: 3,
                            4: 2,
                            5: 4,
                            6: 1,
                            7: 5,
                            8: 6,
                            9: 1,
                            10: 1,
                            11: 7,
                            12: 7,
                            13: 8}

    cluster_mapping = pd.read_pickle(f"../../mcl_toolbox/data/kl_cluster_map.pkl")

    training = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")

    strategy_type_df = pd.DataFrame.from_dict(training)
    strategy_type_df = strategy_type_df.replace(cluster_mapping)
    strategy_type_df = strategy_type_df.replace(cluster_name_mapping)
    strategy_type_df.columns = strategy_type_df.columns.map(str)
    return strategy_type_df


def divide_list_by_indices(sequence, indices):
    result = []
    start_index = 0

    for idx in indices:
        sublist = sequence[start_index:idx]
        result.append(sublist)
        start_index = idx

    # Append the remaining sublist
    sublist = sequence[start_index:]
    result.append(sublist)
    return result


def sequence_improvement(sequences):
    count_sig_increasing = 0
    count_sig_decreasing = 0
    count_increasing = 0
    count_decreasing = 0
    for sequence in sequences:
        result = mk.original_test(sequence)
        if result.trend == "increasing":
            count_sig_increasing += 1
        elif result.trend == "decreasing":
            count_sig_decreasing += 1
        if result.s > 0:
            count_increasing += 1
        elif result.s < 0:
            count_decreasing += 1

    # print(f"{experiment}: Out of {len(sequences)}, {count_sig_increasing} click sequences are increasing, {count_sig_decreasing} are decreasing")
    print(f"{experiment}: {count_increasing} increasing test statistic, {count_decreasing} decreasing test statistic")


def glm(sequences: dict):
    # within strategy analysis on how the clicks changed for the sequence of strategy type
    flattened_values = []
    index_within_list = []
    first_item = []
    pid = []
    variance = []
    cost = []

    def variance_cost(experiment):
        # low variance is 0
        # high cost is 0
        if experiment == "high_variance_high_cost":
            return [1, 1]
        elif experiment == "high_variance_low_cost":
            return [1, 0]
        elif experiment == "low_variance_high_cost":
            return [0, 1]
        elif experiment == "low_variance_low_cost":
            return [0, 0]

    for experiment, value in sequences.items():
        for key, value_list in value.items():
            for i, sublist in enumerate(value_list):
                flattened_values.extend(sublist)
                index_within_list.extend(range(len(sublist)))
                first_item.append([sublist[0]] * len(sublist))
                pid.append([key] * len(sublist))
                variance.append([variance_cost(experiment)[0]] * len(sublist))
                cost.append([variance_cost(experiment)[1]] * len(sublist))

    data = {'number_of_clicks': flattened_values,
            'index_within_list': index_within_list,
            'first_click': [item for sublist in first_item for item in sublist],
            'pid': [item for sublist in pid for item in sublist],
            'variance': [item for sublist in variance for item in sublist],
            'cost': [item for sublist in cost for item in sublist]}

    df = pd.DataFrame(data)

    ### glm
    # formula_ = "number_of_clicks ~ index_within_list*C(condition) + first_click*C(condition)"
    formula_ = "number_of_clicks ~ index_within_list*C(variance)*C(cost) + index_within_list*C(variance) + index_within_list*C(cost) + first_click"
    gamma_model = smf.mixedlm(formula=formula_, data=df, groups=df["pid"]).fit()
    print(gamma_model.summary())


if __name__ == "__main__":
    experiments = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost",
                   "low_variance_high_cost"]
    # experiment = 'low_variance_high_cost'
    all_data = {}
    for experiment in experiments:
        click_df = load_click_data(experiment)
        strategy_type_df = load_strategy_type_data(experiment)

        indices = {}
        for columns in strategy_type_df:
            indices[columns] = get_change_indices(strategy_type_df[columns])

        divided_sequences = {}
        for columns in click_df:
            divided_sequences[columns] = divide_list_by_indices(list(click_df[columns]), indices[columns])

        ### Analysis on sequences
        # sequences = [item for sublist in divided_sequences.values() for item in sublist]
        # sequences = [x for x in sequences if len(x) > 1]
        # sequence_improvement(sequences)


        all_data[experiment] = divided_sequences
    glm(all_data)
