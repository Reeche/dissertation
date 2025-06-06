import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from vars import (
    process_data, process_clicks, learning_participants, clicking_participants,
    assign_model_names, alternative_models, mcrl_models, mb_models
)
import pymannkendall as mk
import statsmodels.formula.api as smf
import warnings
import re
from scipy.stats import shapiro, gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings("ignore")


# ====================== PLOT FUNCTIONS ======================

def plot_confidence_interval(x, pid_average, conf_interval, color, label):
    pid_average = np.array(pid_average, dtype=float)
    plt.plot(pid_average, label=label, color=color, linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color=color, alpha=0.1, label='95% CI')


def calculate_statistics(data_filtered, model_col):
    data_filtered = data_filtered.dropna(subset=[model_col])
    model = np.array(data_filtered[model_col].to_list())
    model_average = np.mean(model, axis=0)
    result = mk.original_test(model_average)
    return model_average, result


def plot_models(data, model_name, model_types, model_col, pid_col, y_limits, ylabel):
    plt.figure(figsize=(8, 6))
    for model in model_types:
        data_filtered = data[data["model"] == model]
        if len(data_filtered) != 0:
            model_average, result = calculate_statistics(data_filtered, model_col)
            model_label = f"{model}: {model_average[0]:.1f} to {model_average[-1]:.1f}"
            plt.plot(model_average, label=model_label)

    data_filtered = data.dropna(subset=[pid_col])
    pid = data_filtered[["pid", pid_col]].drop_duplicates(subset="pid").explode(pid_col).reset_index(drop=True)
    pid["trial"] = pid.groupby("pid").cumcount()
    pid = pid.pivot(index="pid", columns="trial", values=pid_col)
    pid_average = np.mean(pid, axis=0)
    std_err = np.std(pid, axis=0) / np.sqrt(len(pid))
    conf_interval = 1.96 * std_err
    x = np.arange(0, len(pid_average))
    plot_confidence_interval(x, pid_average, conf_interval, "blue", "Participant")

    plt.xlabel("Trial", fontsize=14)
    plt.ylim(y_limits)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=14, ncol=2, loc="lower right")
    plt.savefig(f"plots/{exp}/Participant_{model_name}.png")
    plt.close()


def plot_mer(data, exp):
    data = process_data(data, "model_mer", "pid_mer", exp)
    y_limits = (0, 15) if exp == "c1.1" else (-5, 45) if exp == "c2.1" else (0, 50)
    plot_models(data, "alternatives", alternative_models, "model_mer", "pid_mer", y_limits, "Average maximum expected return")
    plot_models(data, "MF", mcrl_models, "model_mer", "pid_mer", y_limits, "Average maximum expected return")
    plot_models(data, "MB", mb_models, "model_mer", "pid_mer", y_limits, "Average maximum expected return")


def plot_rewards(data, exp):
    data = process_data(data, "model_rewards", "pid_rewards", exp)
    y_limits = (-10, 45) if exp == "c1.1" else (-7, 12)
    if exp == "strategy_discovery":
        mcrl = ["hybrid Reinforce", "MF - Reinforce"]
        plot_models(data, "alternatives", alternative_models, "model_rewards", "pid_rewards", (-100, 10), "Average score")
        plot_models(data, "mcrl", mcrl, "model_rewards", "pid_rewards", (-50, 10), "Average score")
        plot_models(data, "mb", mb_models, "model_rewards", "pid_rewards", (-50, 10), "Average score")


def plot_clicks(data, exp):
    data = process_data(data, "model_clicks", "pid_clicks", exp)
    data["pid_clicks"] = data["pid_clicks"].apply(process_clicks)
    data["model_clicks"] = data["model_clicks"].apply(process_clicks)
    plot_models(data, "alternatives", alternative_models, "model_clicks", "pid_clicks", (0, 12), "Average number of clicks")
    plot_models(data, "mcrl", mcrl_models, "model_clicks", "pid_clicks", (0, 12), "Average number of clicks")
    plot_models(data, "mb", mb_models, "model_clicks", "pid_clicks", (0, 12), "Average number of clicks")


# ====================== STATISTICS FUNCTIONS ======================

def residual_analysis(df, exp, criteria):
    df_filtered = df[[f"model_{criteria}", f"pid_{criteria}", "model"]]
    if criteria == "clicks":
        df_filtered[f"model_{criteria}"] = df_filtered[f"model_{criteria}"].apply(lambda x: [len(i) - 1 for i in ast.literal_eval(x)])
        df_filtered[f"pid_{criteria}"] = df_filtered[f"pid_{criteria}"].apply(lambda x: [len(i) - 1 for i in ast.literal_eval(x)])
    else:
        df_filtered[f"model_{criteria}"] = df_filtered[f"model_{criteria}"].apply(ast.literal_eval)
        df_filtered[f"pid_{criteria}"] = df_filtered[f"pid_{criteria}"].apply(lambda x: [int(num) for num in re.sub(r'[\[\]]', '', x).split()] if isinstance(x, str) else x)

    df_filtered = df_filtered.explode([f"model_{criteria}", f"pid_{criteria}"]).reset_index(drop=True)
    residuals = df_filtered[f"model_{criteria}"] - df_filtered[f"pid_{criteria}"]
    residuals = (residuals - residuals.mean()) / residuals.std()

    x = df_filtered[f"pid_{criteria}"].astype(float).values
    y = residuals.astype(float).values
    z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=z, cmap='viridis', s=80, edgecolor='b', alpha=0.8)
    plt.xlabel("Predicted values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.colorbar(label="Density")
    plt.savefig(f"plots/{exp}/residuals_scatter_all_models_{criteria}_density.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20)
    plt.xlabel("Residuals", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig(f"plots/{exp}/residuals_histogram_{criteria}.png")
    plt.close()

    print("Shapiro-Wilk Test:", shapiro(residuals))


def calculate_model_metrics(df, criteria, exp=None):
    df_filtered = df[[f"model_{criteria}", f"pid_{criteria}", "model"]]
    df_filtered[f"model_{criteria}"] = df_filtered[f"model_{criteria}"].apply(ast.literal_eval)
    df_filtered[f"pid_{criteria}"] = df_filtered[f"pid_{criteria}"].apply(ast.literal_eval)

    df_filtered = df_filtered.explode([f"model_{criteria}", f"pid_{criteria}"]).reset_index(drop=True)
    model_metrics = {}

    for model_type in df_filtered["model"].unique():
        model_df = df_filtered[df_filtered["model"] == model_type]
        y_true = model_df[f"pid_{criteria}"].astype(float)
        y_pred = model_df[f"model_{criteria}"].astype(float)
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        model_metrics[model_type] = {"R²": r2, "RMSE": rmse}
        print(f"Model: {model_type} | R²: {r2:.4f} | RMSE: {rmse:.4f}")

    return model_metrics


# ====================== EXECUTION STARTS HERE ======================

if __name__ == "__main__":
    experiments = ["strategy_discovery"]  # Customize your list
    for exp in experiments:
        print(f"Processing {exp}")
        data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)

        if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
            data = data[data["pid"].isin(clicking_participants[exp])]
        else:
            data = data[data["pid"].isin(learning_participants[exp])]

        data['model'] = data.apply(assign_model_names, axis=1)

        if exp in ["c1.1", "c2.1", "v1.0"]:
            residual_analysis(data, exp, "mer")
        elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]:
            residual_analysis(data, exp, "clicks")
        elif exp == "strategy_discovery":
            residual_analysis(data, exp, "rewards")
