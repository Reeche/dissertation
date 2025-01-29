import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from vars import (process_data, process_clicks,
                  learning_participants, clicking_participants, assign_model_names, alternative_models, mcrl_models, \
                  mb_models)
import pymannkendall as mk
import statsmodels.formula.api as smf
import warnings
import re

# Set the warning filter to "ignore"
warnings.filterwarnings("ignore")


def plot_confidence_interval(x, pid_average, conf_interval, color, label):
    plt.plot(pid_average, label=label, color=color, linewidth=3)

    # transform pid_averages to a numpy array with floats
    pid_average = np.array(pid_average, dtype=float)

    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color=color, alpha=0.1,
                     label='95% CI')


def calculate_statistics(data_filtered, model_col):
    # remove the row in data_filtered[model_col] that has nan values
    data_filtered = data_filtered.dropna(subset=[model_col])
    model = np.array(data_filtered[model_col].to_list())
    model_average = np.mean(model, axis=0)
    result = mk.original_test(model_average)
    return model_average, result


def plot_models(data, model_names, model_col, pid_col, y_limits, ylabel):
    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        data_filtered = data[data["model"] == model_name]
        if len(data_filtered) != 0:
            model_average, result = calculate_statistics(data_filtered, model_col)

            # add first and last trial to the label of the plot
            model_name = f"{model_name}: {model_average[0]:.0f} to {model_average[-1]:.0f}"

            plt.plot(model_average, label=model_name)
            # print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

    # remove the row if the row contains nan values
    data_filtered = data.dropna(subset=[pid_col])

    ### PID data
    # only keep the columns "pid" and "pid_clicks"
    pid = data_filtered[["pid", pid_col]]

    # keep only one row for each unique pid
    pid = pid.drop_duplicates(subset="pid")

    # explode pid_col and add trial number
    pid = pid.explode(pid_col).reset_index(drop=True)
    # add trial number
    pid["trial"] = pid.groupby("pid").cumcount()

    # reshape
    pid = pid.pivot(index="pid", columns="trial", values=pid_col)

    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # transform pid_averages to a numpy array with floats
    pid_average = np.array(pid_average, dtype=float)
    # transform conf_interval to a numpy array with floats
    conf_interval = np.array(conf_interval, dtype=float)

    plot_confidence_interval(x, pid_average, conf_interval, "blue", "Participant")

    plt.xlabel("Trial", fontsize=14)
    plt.ylim(y_limits)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12, ncol=2, loc="lower left")

    plt.savefig(f"plots/{exp}/{model_names}_mer.png")
    plt.show()
    plt.close()


def plot_mer(data, exp):
    data = process_data(data, "model_mer", "pid_mer", exp)

    y_limits = (0, 15) if exp == "c1.1" else (-5, 45) if exp == "c2.1" else (0, 50)
    plot_models(data=data, model_names=alternative_models,  model_col="model_mer",
                pid_col="pid_mer", y_limits=y_limits, ylabel="Average most expected reward")
    plot_models(data=data, model_names=mcrl_models, model_col="model_mer",
                pid_col="pid_mer", y_limits=y_limits, ylabel="Average most expected reward")
    plot_models(data=data, model_names=mb_models, model_col="model_mer",
                pid_col="pid_mer", y_limits=y_limits, ylabel="Average most expected reward")


def plot_rewards(data, exp):
    data = process_data(data, "model_rewards", "pid_rewards", exp)
    # data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])

    if exp == "strategy_discovery":
        y_limits = (-50, 10)
    elif exp == "c1.1":
        y_limits = (-10, 45)
    else:
        y_limits = (-7, 12)

    if exp == "strategy_discovery":
        mcrl_models = ["hybrid Reinforce", "MF - Reinforce"]

    plot_models(data, alternative_models, "alternatives", "model_rewards", "pid_rewards", exp, y_limits,
                "Average score")
    plot_models(data, mcrl_models, "mcrl", "model_rewards", "pid_rewards", exp, y_limits, "Average score")
    plot_models(data, mb_models, "mb", "model_rewards", "pid_rewards", exp, y_limits, "Average score")


def plot_clicks(data, exp):
    data = process_data(data, "model_clicks", "pid_clicks", exp)

    data["pid_clicks"] = data["pid_clicks"].apply(process_clicks)
    data["model_clicks"] = data["model_clicks"].apply(process_clicks)

    plot_models(data, alternative_models, "alternatives", "model_clicks", "pid_clicks", exp,
                (0, 12), "Average number of clicks")
    plot_models(data, mcrl_models, "mcrl", "model_clicks", "pid_clicks", exp, (0, 12),
                "Average number of clicks")
    plot_models(data, mb_models, "mb", "model_clicks", "pid_clicks", exp, (0, 12),
                "Average number of clicks")


def linear_regression(data, exp, criteria="clicks"):
    data = process_data(data, f"model_{criteria}", f"pid_{criteria}", exp)
    if criteria == "clicks":
        data["pid_clicks"] = data["pid_clicks"].apply(process_clicks)
        data["model_clicks"] = data["model_clicks"].apply(process_clicks)

    data = data[["model", "pid", f"model_{criteria}", f"pid_{criteria}"]]

    # for model in data["model"].unique():
    # todo: somehow ols over all models at the same time give insignificant results, whereas individiual model with PID give significant results
    # print(model)
    # data_filtered = data[data["model"] == model]

    # explode together "model_clicks" and "pid_clicks" to get a row for each trial
    data_filtered = data.explode([f'model_{criteria}', f'pid_{criteria}']).reset_index(drop=False)

    # add trials by multiplying range 0 - 35
    if exp != "strategy_discovery":
        data_filtered["trial"] = [i % 35 for i in range(len(data_filtered))]
    else:
        data_filtered["trial"] = [i % 120 for i in range(len(data_filtered))]

    # make sure all columns are integers
    data_filtered[f"model_{criteria}"] = data_filtered[f"model_{criteria}"].apply(lambda x: int(x))
    data_filtered[f"pid_{criteria}"] = data_filtered[f"pid_{criteria}"].apply(lambda x: int(x))
    data_filtered["trial"] = data_filtered["trial"].apply(lambda x: int(x))

    # create one long dataframe for all the clicks of the models with the columns "trial", "model_or_pid", "clicks"
    long_df_model = data_filtered.melt(
        id_vars=["trial", "model"],  # Keep "model" and "trial" as identifiers
        value_vars=[f"model_{criteria}"],  # Columns to unpivot
        var_name="model_pid",  # Temporary column name
        value_name=f"{criteria}"  # Column for the proportions
    )

    # Modify the "model_pid" column to include the model name where appropriate
    long_df_model["model_pid"] = long_df_model.apply(
        lambda row: row["model"] if row["model_pid"] == f"model_{criteria}" else "Not found",
        axis=1
    )
    # rename columns to "trial", "model", "model_or_pid", "clicks"
    long_df_model = long_df_model.rename(columns={"model": "model", "model_pid": "model_pid"})

    # filter data_filtered for unique pid and trial
    data_filtered_pid = data_filtered.drop_duplicates(subset=["pid", "trial"])

    long_df_pid = data_filtered_pid.melt(
        id_vars=["trial", "pid"],  # Keep "model" and "trial" as identifiers
        value_vars=[f"pid_{criteria}"],  # Columns to unpivot
        var_name="model_pid",  # Temporary column name
        value_name=f"{criteria}"  # Column for the proportions
    )

    # rename columns to "trial", "model", "model_or_pid", "clicks"
    long_df_pid = long_df_pid.rename(columns={"pid": "model", "model_pid": "model_pid"})

    # merge the two dataframes
    long_df = pd.concat([long_df_model, long_df_pid])

    results = smf.ols(formula=f"{criteria} ~ C(model_pid, Treatment('pid_{criteria}')) * trial", data=long_df).fit()
    print(results.summary())
    return None


# experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
#               "low_variance_low_cost", "strategy_discovery"]
# experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
experiment = ["v1.0"]

for exp in experiment:
    print(exp)
    df_all = []
    data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)

    if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
        data = data[data["pid"].isin(clicking_participants[exp])]
        # data = data[data["pid"].isin(sd_adaptive_pid)]
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        data = data[data["pid"].isin(learning_participants[exp])]

    # create a new column. If column "class" = "hybrid" and "model_index" = 491, then "model" = "pure Reinforce"
    data['model'] = data.apply(assign_model_names, axis=1)

    if exp in ["c1.1", "c2.1", "v1.0"]:
        plot_mer(data, exp)
        # plot_rewards(data, exp)
        # plot_clicks(data, exp)
        # linear_regression(data, exp, "mer")
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        # plot_rewards(data, exp)
        # plot_clicks(data, exp)
        linear_regression(data, exp, "clicks")
    elif exp in ["strategy_discovery"]:
        plot_rewards(data, exp)
        # plot_clicks(data, exp)
        linear_regression(data, exp, "rewards")

    # plt.ylabel("Performance", fontsize=14)
    # # text size
    # plt.rcParams.update({'font.size': 14})
    # plt.legend()
    # plt.show()
    # plt.close()
