from number_of_click_analysis import create_click_df
import pandas as pd
import pymannkendall as mk

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

if __name__ == "__main__":
    experiments = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost",
                   "low_variance_high_cost"]
    # experiment = 'low_variance_high_cost'

    for experiment in experiments:
        click_df = load_click_data(experiment)
        strategy_type_df = load_strategy_type_data(experiment)

        indices = {}
        for columns in strategy_type_df:
            indices[columns] = get_change_indices(strategy_type_df[columns])

        divided_sequences = []
        for columns in click_df:
            divided_sequences.append(divide_list_by_indices(list(click_df[columns]), indices[columns]))

        sequences = [item for sublist in divided_sequences for item in sublist]
        sequences = [x for x in sequences if len(x) > 1]
        count_increasing = 0
        count_decreasing = 0
        for sequence in sequences:
            result = mk.original_test(sequence)
            if result.trend == "increasing":
                count_increasing += 1
            elif result.trend == "decreasing":
                count_decreasing += 1

        print(f"{experiment}: Out of {len(sequences)}, {count_increasing} click sequences are increasing, {count_decreasing} are decreasing")

