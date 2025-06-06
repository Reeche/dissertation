import pandas as pd
import ast
import sys

from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.global_vars import strategies, features, structure
from mcl_toolbox.utils import learning_utils
from mcl_toolbox.utils.experiment_utils import Experiment


def get_pipeline(exp_name: str, num_trials: int):
    """Returns the environment pipeline for the given experiment."""
    if exp_name not in ["v1.0", "c1.1", "c2.1", "c2.1_dec", "F1", "IRL1"]:
        reward_dist = "categorical"
        reward_structure = exp_name
        reward_distributions = learning_utils.construct_reward_function(
            structure.reward_levels[reward_structure], reward_dist
        )
        repeated_pipeline = learning_utils.construct_repeated_pipeline(
            structure.branchings[exp_name], reward_distributions, num_trials
        )
        return [repeated_pipeline[0] for _ in range(num_trials)]

    pipeline_key = "c2.1_dec" if exp_name == "c2.1" else exp_name
    return [structure.exp_pipelines[pipeline_key][0] for _ in range(num_trials)]


def infer_model_sequences(data, env, exp_name, num_trials, max_evals):
    """Infers planning strategies used by a model from its click sequences."""
    strategy_space = strategies.strategy_space
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights
    normalized_features = learning_utils.get_normalized_features(exp_name)
    weights = learning_utils.get_modified_weights(strategy_space, strategy_weights)
    pipeline = get_pipeline(exp_name, num_trials)

    cm = ComputationalMicroscope(
        pipeline,
        strategy_space,
        weights,
        microscope_features,
        normalized_features=normalized_features,
    )
    return cm.infer_sequences(data, env, max_evals, fit_strategy_temperature=True)


def load_and_prepare_data(exp_name: str, model_name: str) -> pd.DataFrame:
    """Loads and filters the model click data for a given experiment and model."""
    data = pd.read_csv(f"../final_results/aggregated_data/{exp_name}.csv")
    data = data[data['model_index'] == model_name]
    data = data[["pid", "class", "model_index", "model_clicks"]]
    data['model_clicks'] = data['model_clicks'].apply(ast.literal_eval)
    return data


def add_pid_strategies(data: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    """Adds participant-inferred strategies to the DataFrame."""
    pid_strategies = pd.read_pickle(f"../results/cm/inferred_strategies/{exp_name}_training/strategies.pkl")
    data['pid_strategies'] = data['pid'].map(pid_strategies)
    return data


def run_inference_pipeline(exp_name: str, model_name: str, num_trials: int, max_evals: int = 100):
    """Main pipeline for running inference and saving results."""
    data = load_and_prepare_data(exp_name, model_name)
    experiment = Experiment(exp_name, data_path="../results")

    sequences, losses, val1_list, val2_list = [], [], [], []

    for _, row in data.iterrows():
        sequence, loss, val1, val2 = infer_model_sequences(
            row["model_clicks"],
            experiment.participants[row["pid"]].envs,
            exp_name,
            num_trials,
            max_evals
        )
        sequences.append(sequence)
        losses.append(loss)
        val1_list.append(val1)
        val2_list.append(val2)

    data["model_strategies"] = sequences
    data["model_strategies_loss"] = losses
    data["model_strategies_value1"] = val1_list
    data["model_strategies_value2"] = val2_list

    data = add_pid_strategies(data, exp_name)
    data.to_csv(f"{exp_name}_{model_name}_{max_evals}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_model.py <experiment_name> <model_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    model_name = sys.argv[2]
    run_inference_pipeline(exp_name, model_name, num_trials=35)
