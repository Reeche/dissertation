import ast
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from vars import assign_pid_dict, rename_index, rename_map

"""
For the pid best explained by hybrid, mf, habitual, and non-learning models, get the parameter of the fitted model.
Compare the parameter of the fitted model with the models fitted on simulated data.

Hyperparameters of the original fitted models are in final_results/parameters
Hyperparameters of the simulated models are in data
"""


def load_real_data(exp, recovered_model):
    """Loads real data CSV based on experiment and model."""
    files = [file for file in glob.glob(os.path.join("../../final_results/parameters/", "*"))
             if recovered_model.split('_')[0] in os.path.basename(file) and exp in os.path.basename(file)]

    if len(files) > 1:
        raise ValueError(f"Multiple matching files found for {exp} and {recovered_model}: {files}")

    return pd.read_csv(files[0]) if files else None


def load_simulated_data(exp, recovered_model):
    """Loads simulated data CSV based on experiment and model."""
    file_path = f"data/{recovered_model}_{exp}.csv"
    return pd.read_csv(file_path) if os.path.exists(file_path) else None


def extract_parameters(df, parameter):
    """Extracts and converts the parameter of interest from 'model_params'."""
    # remove rows containing nan; these are some missing pid's
    df = df.dropna(subset=["model_params"])
    return df["model_params"].apply(lambda x: ast.literal_eval(x)[parameter])


def process_experiment(exp, recovered_model, comparison_df, parameter, model_indices):
    """Processes a single experiment and model, updating the comparison dataframe."""
    real_data = load_real_data(exp, recovered_model)
    simulated_data = load_simulated_data(exp, recovered_model)

    if real_data is None or simulated_data is None:
        return comparison_df  # Skip if either dataset is missing

    # Filter data based on `pid`
    valid_pids = assign_pid_dict(recovered_model).get(exp, [])
    real_data = real_data[real_data["pid"].isin(valid_pids)]
    simulated_data = simulated_data[simulated_data["pid"].isin(valid_pids)]

    for pid in real_data["pid"].unique():
        row_dict = {"exp": exp, "pid": int(pid), "recovered_model": recovered_model}

        # Extract the parameter value for the real data
        real_data["parameter"] = extract_parameters(real_data, parameter)

        # Extract the parameter value for the simulated data, filtered by pid and model_index
        simulated_data["parameter"] = extract_parameters(simulated_data, parameter)

        # Get the corresponding real parameter value
        row_dict[parameter] = real_data.loc[real_data["pid"] == pid, "parameter"].values[0]

        # Now extract simulated parameters for the specific model indices
        for model_index in model_indices:
            sim_value = simulated_data.loc[
                (simulated_data["pid"] == pid) & (simulated_data["model_index"] == model_index),
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
    Create 4x4 scatter plots comparing different models for the given parameter.

    Args:
    - df: DataFrame containing the parameter values for each model.
    - parameter: The parameter name (e.g., "inverse_temperature") to compare.
    """
    # load csv as df
    df = pd.read_csv(f"results/sd_{parameter}.csv")

    # Extract columns corresponding to the models and the parameter
    models = df['recovered_model'].unique()  # Unique models
    model_columns = [model.strip() for model in models]  # Clean up model names to use in column labels

    # Create a new figure with a 4x4 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))

    for model in model_columns:
        # filter df by model
        df_model = df[df['recovered_model'] == model]
        # df_x are the values in the column with the parameter name, get also the pids
        df_x = df_model[['pid', parameter]].dropna()
        # replace header name by pid and model
        df_x.columns = ['pid', model]

        # iterate over each pair of models
        for i, model_y in enumerate(model_columns):
            # get the column of model_y
            df_y = df_model[['pid', model_y]].dropna()

            ax = axes[model_columns.index(model), i]
            # merge the dataframes on 'pid' to ensure we are comparing the same participants
            merged_df = pd.merge(df_x, df_y, on='pid', suffixes=('_x', '_y'))

            # calculate correlation
            try:
                corr = merged_df[f"{model}_x"].corr(merged_df[f"{model_y}_y"])
            except:
                corr = merged_df[f"{model}"].corr(merged_df[f"{model_y}"])
            print(f"{parameter}: Correlation between {model} and {model_y} is {corr}")

            # calculate Kendall's tau
            try:
                tau = merged_df[f"{model}_x"].corr(merged_df[f"{model_y}_y"], method='kendall')
            except:
                tau = merged_df[f"{model}"].corr(merged_df[f"{model_y}"], method='kendall')
            print(f"{parameter}: Kendall's tau between {model} and {model_y} is {tau}")

            # plot the scatter plot on the corresponding axis if there's valid data
            try:
                ax.scatter(merged_df[f"{model}_x"], merged_df[f"{model_y}_y"], alpha=0.6)
            except:
                ax.scatter(merged_df[f"{model}"], merged_df[f"{model_y}"], alpha=0.6)

            # set labels and title
            ax.set_xlabel(model)
            ax.set_ylabel(model_y)
            ax.set_title(f'{model} vs {model_y}')

            # optionally add a line of identity (y = x) for visual comparison
            try:
                ax.plot([min(merged_df[f"{model}_x"]), max(merged_df[f"{model_y}_y"])],
                    [min(merged_df[f"{model}_x"]), max(merged_df[f"{model_y}_y"])], 'k--', lw=1)
            except:
                ax.plot([min(merged_df[f"{model}"]), max(merged_df[f"{model_y}"])],
                        [min(merged_df[f"{model}"]), max(merged_df[f"{model_y}"])], 'k--', lw=1)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save or show the plot
    plt.savefig(f"sd_{parameter}_scatter_plots.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    ### Define constants inside main block
    # EXPS = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost",
    #         "low_variance_high_cost", "low_variance_low_cost"]
    # RECOVERED_MODELS = ["hybrid_reinforce", "mf_reinforce", "habitual", "non_learning"]
    EXPS = ["c2.1"]
    RECOVERED_MODELS = ["hybrid_reinforce", "mf_reinforce", "habitual", "non_learning"]
    PARAMETERS_OF_INTEREST = ["gamma", "lr", "inverse_temperature"]
    MODEL_INDICES = [491, 3326, 1743, 1756]

    ### Create the csv
    for parameter in PARAMETERS_OF_INTEREST:
        comparison_df = pd.DataFrame(columns=["exp", "pid", parameter, "recovered_model", *MODEL_INDICES])
        for exp in EXPS:
            for recovered_model in RECOVERED_MODELS:
                comparison_df = process_experiment(exp, recovered_model, comparison_df, parameter,
                                                   MODEL_INDICES)

        # save csv after renaming
        comparison_df = rename(comparison_df)
        comparison_df.to_csv(f"results/sd_{parameter}.csv", index=False)

    ### Create scatter plots
    # for parameter in PARAMETERS_OF_INTEREST:
    #     create_scatter_plots(parameter)
