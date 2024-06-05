import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from vars import learning_participants, clicking_participants, assign_model_names
import pymannkendall as mk
import statsmodels.formula.api as smf
import warnings

# Set the warning filter to "ignore"
warnings.filterwarnings("ignore")


def plot_mer(data, exp):
    # get the model_mer and pid_mer
    data = data[["model", "model_mer", "pid_mer"]]
    data["pid_mer"] = data["pid_mer"].apply(lambda x: ast.literal_eval(x))
    data["model_mer"] = data["model_mer"].apply(lambda x: ast.literal_eval(x))

    for model_name in ["Non-learning", "SSL", "Habitual"]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_mer"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

    ### PID data
    pid = np.array(data_filtered["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average most expected reward", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/alternatives_mer.png")
    # plt.show()
    plt.close()

    for model_name in ["hybrid LVOC", "hybrid Reinforce", "MF - LVOC", "MF - Reinforce"]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_mer"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

    ### PID data
    pid = np.array(data_filtered["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average most expected reward", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/MF_mer.png")
    # plt.show()
    plt.close()

    for model_name in ["MB - No assump., grouped", "MB - No assump., ind.",
                       "MB - Uniform, ind.", "MB - Uniform, grouped",
                       "MB - Level, grouped", "MB - Level, ind."]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_mer"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

    ### PID data
    pid = np.array(data_filtered["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1)

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average most expected reward", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/MB_mer.png")
    # plt.show()
    plt.close()


def plot_rewards(data):
    # get the model_mer and pid_mer
    data = data[["model", "model_rewards", "pid_rewards"]]
    # get unique model names
    # model_names = data["model"].unique()

    # data["pid_rewards"] = data["pid_rewards"].apply(lambda x: list(ast.literal_eval(x)))
    data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])

    data["model_rewards"] = data["model_rewards"].apply(lambda x: ast.literal_eval(x))

    for model_name in ["Non-learning", "SSL", "Habitual"]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_rewards"].to_list())
        model_average = np.mean(model, axis=0)
        # plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_rewards(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_rewards"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average score", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/pid_rewards.png")
    # plt.show()
    plt.close()

    for model_name in ["hybrid LVOC", "hybrid Reinforce", "MF - LVOC", "MF - Reinforce"]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_rewards"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_rewards(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_rewards"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average score", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/MF_rewards.png")
    # plt.show()
    plt.close()

    for model_name in ["MB - No assump., grouped", "MB - No assump., ind.",
                       "MB - Uniform, ind.", "MB - Uniform, grouped",
                       "MB - Level, grouped", "MB - Level, ind."]:
        data_filtered = data[data["model"] == model_name]
        model = np.array(data_filtered["model_rewards"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_rewards(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_rewards"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1)

    plt.xlabel("Trial", fontsize=12)
    # plt.ylim(0, 12)
    plt.ylabel("Average score", fontsize=12)
    plt.legend()
    plt.savefig(f"plots/{exp}/MB_rewards.png")
    # plt.show()
    plt.close()

def compare_model_to_pid_clicks(model_name, data):
    # how much variability in the pid data is explained by the model
    # explode together "model_clicks" and "pid_clicks" to get a row for each trial
    data_ = data.explode(['model_clicks', 'pid_clicks']).reset_index(drop=False)

    # make sure all columns are integers
    data_["model_clicks"] = data_["model_clicks"].apply(lambda x: int(x))
    data_["pid_clicks"] = data_["pid_clicks"].apply(lambda x: int(x))
    data_["index"] = data_["index"].apply(lambda x: int(x))

    results = smf.ols(formula="pid_clicks ~ index*model_clicks", data=data_).fit()  # Fit model for first regression line
    print(model_name)
    # print(results.summary())
    r_squared = results.rsquared
    print("R-squared:", r_squared)
    print("-" * 50)
    return None

def compare_model_to_pid_rewards(model_name, data):
    # explode together "model_clicks" and "pid_clicks" to get a row for each trial
    data_ = data.explode(['model_rewards', 'pid_rewards']).reset_index(drop=False)

    # make sure all columns are integers
    data_["model_rewards"] = data_["model_rewards"].apply(lambda x: int(x))
    data_["pid_rewards"] = data_["pid_rewards"].apply(lambda x: int(x))
    data_["index"] = data_["index"].apply(lambda x: int(x))

    # results = smf.ols(formula="pid_rewards ~ index*model_rewards", data=data_).fit()  # Fit model for first regression line
    # print(model_name)
    # print(results.summary())
    # r_squared = results.rsquared
    # print("R-squared:", r_squared)
    # print("-" * 50)
    return None

def plot_clicks(data):
    data = data[["model", "model_clicks", "pid_clicks"]]
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: ast.literal_eval(x))

    data["model_clicks"] = data["model_clicks"].apply(lambda x: ast.literal_eval(x))

    plt.figure(figsize=(8, 6))

    for model_name in ["Non-learning", "SSL", "Habitual"]:
        data_filtered = data[data["model"] == model_name]
        lengths_model = []
        lengths_pid = []
        # Iterate through the DataFrame
        for index, row in data_filtered.iterrows():
            lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
            lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
        data_filtered["model_clicks"] = lengths_model
        data_filtered["pid_clicks"] = lengths_pid

        model = np.array(data_filtered["model_clicks"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_clicks(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_clicks"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    plt.ylim(0, 12)
    plt.ylabel("Average number of clicks", fontsize=12)
    plt.legend(fontsize=12, ncol=2)
    plt.savefig(f"plots/{exp}/alternatives_clicks.png")
    # plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))

    for model_name in ["hybrid LVOC", "hybrid Reinforce", "MF - LVOC", "MF - Reinforce"]:
        data_filtered = data[data["model"] == model_name]
        lengths_model = []
        lengths_pid = []
        # Iterate through the DataFrame
        for index, row in data_filtered.iterrows():
            lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
            lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
        data_filtered["model_clicks"] = lengths_model
        data_filtered["pid_clicks"] = lengths_pid

        model = np.array(data_filtered["model_clicks"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_clicks(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_clicks"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.xlabel("Trial", fontsize=12)
    plt.ylim(0, 12)
    plt.ylabel("Average number of clicks", fontsize=12)
    plt.legend(fontsize=12, ncol=2)
    plt.savefig(f"plots/{exp}/MF_clicks.png")
    # plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))

    for model_name in ["MB - No assump., grouped", "MB - No assump., ind.",
                       "MB - Uniform, ind.", "MB - Uniform, grouped",
                       "MB - Level, grouped", "MB - Level, ind."]:
        data_filtered = data[data["model"] == model_name]
        lengths_model = []
        lengths_pid = []
        # Iterate through the DataFrame
        for index, row in data_filtered.iterrows():
            lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
            lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
        data_filtered["model_clicks"] = lengths_model
        data_filtered["pid_clicks"] = lengths_pid

        model = np.array(data_filtered["model_clicks"].to_list())
        model_average = np.mean(model, axis=0)
        plt.plot(model_average, label=model_name)

        result = mk.original_test(model_average)
        print(f"{model_name}: trend={result[0]}, p={result[2]}, statistic={result[5]}")

        # compare_model_to_pid_clicks(model_name, data_filtered)

    ### PID data
    pid = np.array(data_filtered["pid_clicks"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(pid_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1)

    plt.xlabel("Trial", fontsize=12)
    plt.ylim(0, 12)
    plt.ylabel("Average number of clicks", fontsize=12)
    plt.legend(fontsize=12, ncol=2)
    plt.savefig(f"plots/{exp}/MB_clicks.png")
    # plt.show()
    plt.close()


# experiment = ["v1.0", "c2.1", "c1.1"]
# experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
#               "low_variance_low_cost", "strategy_discovery"]
experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
              "low_variance_low_cost"]
# experiment = ["high_variance_low_cost"]

for exp in experiment:
    print(exp)
    df_all = []
    data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)

    if exp in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
        data = data[data["pid"].isin(clicking_participants[exp])]
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        data = data[data["pid"].isin(learning_participants[exp])]

    ### for debugging filter for model-based models
    # data = data[data["model_index"] == "full"]

    # create a new column. If column "class" = "hybrid" and "model_index" = 491, then "model" = "pure Reinforce"
    data['model'] = data.apply(assign_model_names, axis=1)

    # filter for hybrid Reinforce and SSL models
    # data = data[dmerata["model"].isin(["hybrid Reinforce"])]

    if exp in ["c1.1", "c2.1", "v1.0", "strategy_discovery"]:
        # plot_mer(data, exp)
        plot_rewards(data)
        # plot_clicks(data)
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        # plot_rewards(data)
        plot_clicks(data)

    # plt.ylabel("Performance", fontsize=14)
    # # text size
    # plt.rcParams.update({'font.size': 14})
    # plt.legend()
    # plt.show()
    # plt.close()
