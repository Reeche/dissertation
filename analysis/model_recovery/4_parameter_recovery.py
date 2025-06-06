import ast
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from vars import assign_pid_dict, rename_index, rename_map, pid_dict

import warnings
warnings.filterwarnings("ignore")

"""
For the pid best explained by hybrid, mf, habitual, and non-learning models, get the parameter of the fitted model.
Compare the parameter of the fitted model with the models fitted on simulated data.

Hyperparameters of the original fitted models are in final_results/parameters
Hyperparameters of the simulated models are in data
"""


def load_real_data(exp, recovered_model):
    """Loads real data CSV based on experiment and model."""
    if recovered_model.startswith("variant/"):
        files = [file for file in glob.glob(os.path.join("../reinforce_variants_comparison/parameters/", "*"))
                 if exp in os.path.basename(file)]
    else:
        files = [file for file in glob.glob(os.path.join("../../final_results/parameters/", "*"))
                 if recovered_model.split('_')[0] in os.path.basename(file) and exp in os.path.basename(file)]

    if len(files) > 1:
        raise ValueError(f"Multiple matching files found for {exp} and {recovered_model}: {files}")

    return pd.read_csv(files[0]) if files else None


def load_simulated_data(exp, recovered_model):
    """Loads simulated data CSV based on experiment and model."""
    if recovered_model.startswith("variant/"):
        file_path = f"../model_recovery/data/variants/{recovered_model.split('/')[1]}_{exp}.csv"
    else:
        file_path = f"data/{recovered_model}_{exp}.csv"
    return pd.read_csv(file_path) if os.path.exists(file_path) else None


def extract_parameters(df, parameter):
    """Extracts and converts the parameter of interest from 'model_params'."""
    # remove rows containing nan; these are some missing pid's
    df = df.dropna(subset=["model_params"])
    df["model_params"] = df["model_params"].apply(lambda x: ast.literal_eval(x))
    df["model_params"] = df["model_params"].apply(lambda x: x.get(parameter))
    return df["model_params"]


def process_experiment(exp, recovered_model, comparison_df, parameter, model_indices):
    """Processes a single experiment and model, updating the comparison dataframe."""
    real_data = load_real_data(exp, recovered_model)
    simulated_data = load_simulated_data(exp, recovered_model)

    if parameter == "subjective_cost":
        # filter for model_indices and remove the models that are not in the list
        simulated_data = simulated_data[simulated_data["model_index"].isin(model_indices)]
        # remove the models that are not in the list
        real_data = real_data[real_data["model"].isin(model_indices)]

    if real_data is None or simulated_data is None:
        return comparison_df  # Skip if either dataset is missing

    # rename real_data "parameters" column to "model_params"
    if real_data is not None:
        real_data = real_data.rename(columns={"parameters": "model_params"})

    # Filter data based on `pid`
    if recovered_model.startswith("variant/"):
        recovered_model = recovered_model.split("/")[1]
        valid_pids = pid_dict[exp]
    else:
        valid_pids = assign_pid_dict(recovered_model).get(exp, [])

    real_data = real_data[real_data["pid"].isin(valid_pids)]
    simulated_data = simulated_data[simulated_data["pid"].isin(valid_pids)]

    for pid in real_data["pid"].unique():
        row_dict = {"exp": exp, "pid": int(pid), "recovered_model": recovered_model}

        # filter real_data and simulated_data by pid
        real_data_ = real_data[real_data["pid"] == pid]
        simulated_data_ = simulated_data[simulated_data["pid"] == pid]

        # Extract the parameter value for the real data
        real_data_["parameter"] = extract_parameters(real_data_, parameter)

        # Extract the parameter value for the simulated data, filtered by pid and model_index
        simulated_data_["parameter"] = extract_parameters(simulated_data_, parameter)

        # Get the corresponding real parameter value
        row_dict[parameter] = real_data_.loc[real_data_["pid"] == pid, "parameter"].values[0]

        # Now extract simulated parameters for the specific model indices
        for model_index in model_indices:
            sim_value = simulated_data_.loc[
                (simulated_data_["pid"] == pid) & (simulated_data_["model_index"] == model_index),
                "parameter"
            ].values[0]  # Extract the value for the specific parameter
            row_dict[model_index] = sim_value

        # Append the row to the DataFrame
        comparison_df = pd.concat([comparison_df, pd.DataFrame([row_dict])], ignore_index=True)

    return comparison_df


def rename(df):
    # rename the header names using rename_index
    df = df.rename(columns=rename_index)
    # rename the recovered_model column using rename_map
    df["recovered_model"] = df["recovered_model"].replace(rename_map)
    return df


def create_scatter_plots(parameter):
    """
    For each model, create a scatter plot comparing the true parameter vs. recovered parameter.

    Args:
    - parameter: The parameter name (e.g., "inverse_temperature") to compare.
    """

    # Load CSV as DataFrame
    # df = pd.read_csv(f"results/variants/{parameter}.csv")
    df = pd.read_csv(f"results/sd_{parameter}.csv")

    # For variants only: replace recovered_model with the rename_index mapping (assumed defined elsewhere)
    df["recovered_model"] = df["recovered_model"].replace(rename_index)

    # Get unique models
    models = df['recovered_model'].unique()

    for model in models:
        # Filter for rows where recovered_model == model
        df_model = df[df['recovered_model'] == model]

        # Check if model column exists
        if model not in df_model.columns:
            print(f"Skipping model '{model}' — no recovered parameter column found.")
            continue

        # Extract true and recovered parameter values
        df_plot = df_model[['pid', parameter, model]].dropna()

        if df_plot.empty:
            print(f"No valid data for model '{model}'. Skipping.")
            continue

        true_values = df_plot[parameter]
        recovered_values = df_plot[model]

        # Calculate correlations
        pearson_corr = true_values.corr(recovered_values)
        kendall_tau = true_values.corr(recovered_values, method='kendall')

        print(f"{parameter} — Model: {model}")
        print(f"Pearson correlation: {pearson_corr:.3f}")
        print(f"Kendall's tau: {kendall_tau:.3f}")

        # Plot scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(true_values, recovered_values, alpha=0.6)
        plt.xlabel('True parameter')
        plt.ylabel('Recovered parameter')
        plt.title(f'{model}: {parameter}\nPearson r={pearson_corr:.3f}, Kendall tau={kendall_tau:.3f}')

        # Line of identity (y = x)
        min_val = min(true_values.min(), recovered_values.min())
        max_val = max(true_values.max(), recovered_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

        plt.tight_layout()
        plt.savefig(f"results/plots/{parameter}_{model}.png", dpi=300)
        # plt.show()
        plt.close()


if __name__ == "__main__":
    ### Define constants inside main block
    EXPS = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost",
            "low_variance_high_cost", "low_variance_low_cost"]

    RECOVERED_MODELS = ["hybrid_reinforce", "mf_reinforce", "habitual", "non_learning"]
    PARAMETERS_OF_INTEREST = ["gamma", "lr", "inverse_temperature"]
    MODEL_INDICES = [491, 3326, 1743, 1756]


    ### Create the csv
    for parameter in PARAMETERS_OF_INTEREST:

        if parameter == "subjective_cost":
            MODEL_INDICES = [3315, 3316, 3323, 3324]
        else:
            MODEL_INDICES = [3315, 3316, 3317, 3318, 3323, 3324, 3325, 3326]

        comparison_df = pd.DataFrame(columns=["exp", "pid", parameter, "recovered_model", *MODEL_INDICES])

        for exp in EXPS:
            for recovered_model in RECOVERED_MODELS:
                comparison_df = process_experiment(exp, recovered_model, comparison_df, parameter,
                                                   MODEL_INDICES)

        # save csv after renaming
        comparison_df = rename(comparison_df)

        comparison_df.to_csv(f"{parameter}.csv", index=False)

    ### Create scatter plots
    for parameter in PARAMETERS_OF_INTEREST:
        create_scatter_plots(parameter)
