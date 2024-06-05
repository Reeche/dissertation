import pandas as pd
import ast
import sys

from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.global_vars import strategies, features, structure, assign_model_names
from mcl_toolbox.utils import learning_utils
from mcl_toolbox.utils.experiment_utils import Experiment

"""
Infer the planning strategy used by the model via their click sequences

"""


def infer_model_sequences(data, env, num_trials, max_evals):
    # 79 strategies out of 89
    strategy_space = strategies.strategy_space
    # no habitual features because each trial is considered individually
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    # For the new experiment that are not either v1.0, c1.1, c2.1_dec, F1 or IRL1
    if exp_name not in ["v1.0", "c1.1", "c2.1", "c2.1_dec", "F1", "IRL1"]:
        reward_dist = "categorical"
        reward_structure = exp_name
        reward_distributions = learning_utils.construct_reward_function(
            structure.reward_levels[reward_structure], reward_dist
        )
        repeated_pipeline = learning_utils.construct_repeated_pipeline(
            structure.branchings[exp_name], reward_distributions, num_trials
        )
        exp_pipelines = {exp_name: repeated_pipeline}
    else:
        # list of all experiments, e.g. v1.0, T1.1 only has the transfer after training (20 trials)
        exp_pipelines = structure.exp_pipelines
    #     if exp_name not in structure.exp_reward_structures:
    #         raise (ValueError, "Reward structure not found.")
    #     # reward_structure = structure.exp_reward_structures[exp_name]
    #
    # if exp_name not in exp_pipelines:
    #     raise (ValueError, "Experiment pipeline not found.")

    if exp_name == "c2.1":
        pipeline = exp_pipelines["c2.1_dec"]
    else:
        pipeline = exp_pipelines[exp_name]
    pipeline = [pipeline[0] for _ in range(num_trials)]

    normalized_features = learning_utils.get_normalized_features(exp_name)
    W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(
        pipeline,
        strategy_space,
        W,
        microscope_features,
        normalized_features=normalized_features,
    )

    inferred_sequence = cm.infer_sequences(data, env, max_evals, fit_strategy_temperature=True)
    return inferred_sequence[0], inferred_sequence[1], inferred_sequence[2], inferred_sequence[3]


if __name__ == "__main__":
    exp_name = str(sys.argv[1])
    model_name = str(sys.argv[2])

    # exp_name = "low_variance_high_cost"
    # model_name = "level_level"

    number_of_trials = 35

    ## Get the data
    data = pd.read_csv(f"../final_results/aggregated_data/{exp_name}.csv")
    # filter by model_name
    data = data[data['model_index'] == model_name]
    # assign model names
    # data['model'] = data.apply(assign_model_names, axis=1)

    # drop irrelevant information
    data = data[["pid", "class", "model_index", "model_clicks"]]
    # get model clicks
    data['model_clicks'] = data['model_clicks'].apply(ast.literal_eval)

    E = Experiment(exp_name, data_path=f"../results")

    sequence_list = []
    loss_list = []
    some_value_list = []  # not sure what value is it or for what it is used
    some_value2_list = []  # not sure what value is it or for what it is used
    for idx, row in data.iterrows():
        inferred_strategies, loss, value1, value2 = infer_model_sequences(row["model_clicks"],
                                                                          E.participants[row["pid"]].envs,
                                                                          number_of_trials, max_evals=100)
        sequence_list.append(inferred_strategies)
        loss_list.append(loss)
        some_value_list.append(value1)
        some_value2_list.append(value2)

    # some value are some values that are returned by the CM but I do not know what they are, so saved just in case they are needed
    data["model_strategies"] = sequence_list
    data["model_strategies_loss"] = loss_list
    data["model_strategies_value1"] = some_value_list
    data["model_strategies_value2"] = some_value2_list

    # append the participant strategies
    pid_strategies = pd.read_pickle(f"../results/cm/inferred_strategies/{exp_name}_training/strategies.pkl")
    data['pid_strategies'] = data['pid'].map(lambda x: pid_strategies[x])

    # save as csv
    data.to_csv(f"{exp_name}_{model_name}_100.csv", index=False)
