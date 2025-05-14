import pandas as pd
import pymannkendall as mk
import statsmodels.formula.api as smf
import statsmodels.api as sm


def get_change_indices(numbers):
    change_indices = []

    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1]:
            change_indices.append(i)

    return change_indices


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


def linear_mixed_effect_model(score, type):
    # if testing for only one condition
    del score['v1.0'], score['c2.1']
    del type['v1.0'], type['c2.1']

    # within strategy analysis on how the clicks changed for the sequence of strategy type
    flattened_values = []
    index_within_list = []
    first_score = []
    pid = []
    condition = []
    strategy_type = []
    improved_score = []

    def which_cond(exp):
        if exp == "v1.0":
            return 0
        if exp == "c2.1":
            return 1
        if exp == "c1.1":
            return 2

    for experiment, value in score.items():
        for pid_, score_list in value.items():
            for i, sublist in enumerate(score_list):
                flattened_values.extend(sublist)
                index_within_list.extend(range(len(sublist)))
                first_score.append([sublist[0]] * len(sublist))
                pid.append([pid_] * len(sublist))
                condition.append([which_cond(experiment)] * len(sublist))
                for score in sublist:
                    improved_score.append(score - 28.55)  # 28.55 for decreasing variance

    for experiment, value in type.items():
        for pid_, type_list in value.items():
            for i, sublist in enumerate(type_list):
                strategy_type.extend(sublist)

    data_ = {'score': flattened_values,
             'index_within_list': index_within_list,
             'first_score': [item for sublist in first_score for item in sublist],
             'pid': [item for sublist in pid for item in sublist],
             'condition': [item for sublist in condition for item in sublist],
             'strategy_type': strategy_type,
             'improved_score': improved_score}

    df = pd.DataFrame(data_)

    ### glm
    # formula_ = "score ~ index_within_list*C(condition)"
    # # formula_ = "improved_score ~ index_within_list"
    # gamma_model = smf.mixedlm(formula=formula_, data=df, groups=df["pid"]).fit()
    # print(gamma_model.summary())

    # testing glm for every strategy type
    for type in list(set(strategy_type)):
        data = df[df["strategy_type"] == type]
        ### glm
        formula_ = "score ~ index_within_list"
        try:
            lme_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
            ### ordinary least square
            # y = data["score"]
            # X = data["index_within_list"]
            # X = sm.add_constant(X)
            # ols_model = sm.OLS(y, X).fit()
            print("Strategy type: ", type)
            print(lme_model.summary())
        except:
            print("skipped type", type)

    return None


def strategy_type_mapping(data):
    # add cluster names
    cluster_name_mapping = {1: "Goal-setting with exhaustive backward planning",
                            2: "Forward planning strategies similar to Breadth First Search",
                            3: "Middle-out planning",
                            4: "Forward planning strategies similar to Best First Search",
                            5: "Local search",
                            6: "Maximizing Goal-setting with exhaustive backward planning",
                            7: "Frugal planning",
                            8: "Myopic planning",
                            9: "Maximizing goal-setting with limited backward planning",
                            10: "Frugal goal-setting strategies",
                            11: "Strategy that explores immediate outcomes on the paths to the best final outcomes",
                            12: "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing",
                            13: "Miscellaneous strategies"}

    clustering_the_cluster = {"Goal-setting with exhaustive backward planning": "Goal-setting",
                              "Forward planning strategies similar to Breadth First Search": "Forward planning",
                              "Middle-out planning": "Middle-out planning",
                              "Forward planning strategies similar to Best First Search": "Forward planning",
                              "Local search": "Local search",
                              "Maximizing Goal-setting with exhaustive backward planning": "Goal-setting",
                              "Frugal planning": "Frugal planning",
                              "Myopic planning": "Little planning",
                              "Maximizing goal-setting with limited backward planning": "Goal-setting",
                              "Frugal goal-setting strategies": "Goal-setting",
                              "Strategy that explores immediate outcomes on the paths to the best final outcomes": "Final and then immediate outcome",
                              "Strategy that explores immediate outcomes on the paths to the best final outcomes with satisficing": "Final and then immediate outcome",
                              "Miscellaneous strategies": "Miscellaneous strategies"}

    replace_by_numbers = {
        "Goal-setting": 1,
        "Forward planning": 2,
        "Middle-out planning": 3,
        "Local search": 4,
        "Frugal planning": 5,
        "Little planning": 6,
        "Final and then immediate outcome": 7,
        "Miscellaneous strategies": 8,
    }
    # strategy type mapping
    # map strategy to cluster
    cluster_mapping = pd.read_pickle(f"../../mcl_toolbox/data/kl_cluster_map.pkl")
    strategy_cluster_df = data.replace(cluster_mapping)
    strategy_cluster_df = strategy_cluster_df.replace(cluster_name_mapping)
    strategy_cluster_df = strategy_cluster_df.replace(clustering_the_cluster)
    strategy_cluster_cluster_df = strategy_cluster_df.replace(replace_by_numbers)

    return strategy_cluster_cluster_df


def remove_optimally_starting_sequences(all_scores, all_types):
    filtered_scores = {}
    filtered_types = {}

    for exp, values in all_types.items():
        if exp == "v1.0":
            adaptive_types = [1]
        else:
            adaptive_types = [2]

        # Create new dictionaries to store the filtered items
        filtered_dict_types = {}
        filtered_dict_scores = {}

        # Iterate through dict_a items (key-value pairs)
        for key, value in values.items():
            filtered_type_list = []
            filtered_score_list = []

            # Iterate through the lists in the value
            for sublist_index, sublist in enumerate(value):
                if sublist[0] not in adaptive_types:
                    filtered_type_list.append(sublist)
                    filtered_score_list.append(all_scores[exp][key][sublist_index])

            # Add the filtered lists to the new dictionaries
            filtered_dict_types[key] = filtered_type_list
            filtered_dict_scores[key] = filtered_score_list

        # Replace the original dictionaries with the filtered dictionaries
        filtered_types[exp] = filtered_dict_types
        filtered_scores[exp] = filtered_dict_scores

    return filtered_types, filtered_scores


if __name__ == "__main__":
    exp_list = ["v1.0", "c2.1", "c1.1"]
    all_scores = {}
    all_types = {}

    participants_starting_optimally = {
        "v1.0": [85, 140],
        "c2.1": [0, 8, 39, 41, 58, 72, 99, 130, 152, 172],
        "c1.1": [2, 7, 36, 37, 42, 89, 91, 92, 157, 168]
    }

    participants_clicked_nothing = {
        "v1.0": [51, 141],
        "c2.1": [3, 11],
        "c1.1": [46, 87, 97]
    }

    for exp in exp_list:
        # load fitted strategies
        strategy_df = pd.DataFrame.from_dict(
            pd.read_pickle(f"../../results/cm/inferred_strategies/{exp}_training/strategies.pkl"))

        ## remove some participants
        # strategy_df = strategy_df.drop(columns=participants_starting_optimally[exp])

        # load strategy score
        score_mapping = pd.read_pickle(f"../../results/cm/strategy_scores/{exp}_strategy_scores.pkl")

        # create df with mapped strategy score
        # score_mapping start from 0 but inferred_strategies.pkl start from 1
        strategy_score_df = strategy_df - 1

        # replace strategy with score
        strategy_score_df = strategy_score_df.replace(score_mapping)

        # create mapped strategy types for each pid
        mapped_strategy_type = strategy_type_mapping(strategy_df)

        # get the index of the strategy changes
        indices = {}
        for columns in strategy_df:
            indices[columns] = get_change_indices(strategy_df[columns])

        divided_sequences_score = {}
        for columns in strategy_score_df:
            divided_sequences_score[columns] = divide_list_by_indices(list(strategy_score_df[columns]),
                                                                      indices[columns])

        divided_sequences_type = {}
        for columns in strategy_score_df:
            divided_sequences_type[columns] = divide_list_by_indices(list(mapped_strategy_type[columns]),
                                                                     indices[columns])

        ### Analysis on sequences
        # sequences = [item for sublist in divided_sequences.values() for item in sublist]
        # sequences = [x for x in sequences if len(x) > 1]

        all_scores[exp] = divided_sequences_score
        all_types[exp] = divided_sequences_type

    all_types, all_scores = remove_optimally_starting_sequences(all_scores, all_types)
    linear_mixed_effect_model(all_scores, all_types)
