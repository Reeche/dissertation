import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from vars import learning_participants, clicking_participants
import pymannkendall as mk


def plot_mer(data, exp):
    # get the model_mer and pid_mer
    data = data[["model", "model_mer", "pid_mer"]]
    data["pid_mer"] = data["pid_mer"].apply(lambda x: ast.literal_eval(x))
    data["model_mer"] = data["model_mer"].apply(lambda x: ast.literal_eval(x))

    # get unique model names
    model_names = data["model"].unique()

    for model_name in model_names:
        data_filtered = data[data["model"] == model_name]
        # convert series to np array
        model = np.array(data_filtered["model_mer"].to_list())
        model_average = np.mean(model, axis=0)


        x = np.arange(0, len(model_average))

        # plot model_mer and pid_mer


        # plt.plot(model_average, label=model_name, color="orange")
        if model_name == "hybrid Reinforce":
            plt.plot(model_average, label="MCRL", linewidth=3)
        else:
            plt.plot(model_average, label=model_name, linewidth=3)

    pid = np.array(data_filtered["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)  # todo:check
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')
    plt.plot(pid_average, label="Participants", color="blue", linewidth=3)


    plt.xlabel("Trial", fontsize=14)
    if exp in ["c1.1"]:
        plt.ylim(5, 15)
    elif exp in ["c2.1"]:
        plt.ylim(10, 50)
    elif exp in ["v1.0"]:
        plt.ylim(10, 50)

        # plt.ylabel("Average expected score")
        # plt.legend()
        # plt.savefig(f"plots/{exp}/{model_name}_mer.png")
        # plt.show()
        # plt.close()


def plot_rewards(data):
    # get the model_mer and pid_mer
    data = data[["model", "model_rewards", "pid_rewards"]]
    # get unique model names
    model_names = data["model"].unique()

    # data["pid_rewards"] = data["pid_rewards"].apply(lambda x: list(ast.literal_eval(x)))
    data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])

    data["model_rewards"] = data["model_rewards"].apply(lambda x: ast.literal_eval(x))

    for model_name in model_names:
        data_filtered = data[data["model"] == model_name]
        # get the mean
        model = np.array(data_filtered["model_rewards"].to_list())
        model_average = np.mean(model, axis=0)
        pid = np.array(data_filtered["pid_rewards"].to_list())
        pid_average = np.mean(pid, axis=0)

        # Mann Kendall test of trend
        result = mk.original_test(model_average)
        print(model_name, result)

        # Calculate mean and standard error for each data point
        std_dev = np.std(pid, axis=0)
        n = len(pid)  # todo:check
        std_err = std_dev / np.sqrt(n)

        # Calculate the confidence interval
        conf_interval = 1.96 * std_err

        x = np.arange(0, len(model_average))

        # plot model_mer and pid_mer
        plt.plot(pid_average, label="Participant", color="blue")
        plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                         label='95% CI')

        plt.plot(model_average, label=model_name, color="orange")

        plt.xlabel("Trial", fontsize=12)
        # plt.ylim(0, 50)
        plt.ylabel("Average score")
        plt.legend()
        plt.savefig(f"plots/{exp}/{model_name}_rewards.png")

        # plt.show()
        plt.close()


def plot_clicks(data):
    data = data[["model", "model_clicks", "pid_clicks"]]
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: ast.literal_eval(x))

    data["model_clicks"] = data["model_clicks"].apply(lambda x: ast.literal_eval(x))

    # get unique model names
    model_names = data["model"].unique()

    for model_name in model_names:
        data_filtered = data[data["model"] == model_name]
        lengths_model = []
        lengths_pid = []
        # Iterate through the DataFrame
        for index, row in data_filtered.iterrows():
            lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
            lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
        data_filtered["model_clicks"] = lengths_model
        data_filtered["pid_clicks"] = lengths_pid

        # convert series to np array
        model = np.array(data_filtered["model_clicks"].to_list())
        model_average = np.mean(model, axis=0)
        pid = np.array(data_filtered["pid_clicks"].to_list())
        pid_average = np.mean(pid, axis=0)

        # Calculate mean and standard error for each data point
        std_dev = np.std(pid, axis=0)
        n = len(pid)
        std_err = std_dev / np.sqrt(n)

        # Calculate the confidence interval
        conf_interval = 1.96 * std_err

        x = np.arange(0, len(model_average))

        # plot model_mer and pid_mer
        plt.plot(pid_average, label="Participant", color="blue")
        plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                         label='95% CI')

        plt.plot(model_average, label=model_name, color="orange")

        plt.xlabel("Trial", fontsize=12)
        plt.ylim(2, 10)
        plt.ylabel("Average clicks", fontsize=12)
        plt.legend(fontsize=12)
        # x, y axis font size
        plt.rcParams.update({'font.size': 12})
        plt.savefig(f"plots/{exp}/{model_name}_clicks.png")
        # plt.show()
        plt.close()


def assign_model_names(row):
    if row['class'] == 'hybrid' and row['model_index'] == "491":
        return 'hybrid Reinforce'
    elif row['class'] == 'hybrid' and row['model_index'] == "479":
        return 'hybrid LVOC'
    elif row['class'] == 'pure' and row['model_index'] == "491":
        return 'pure Reinforce'
    elif row['class'] == 'pure' and row['model_index'] == "479":
        return 'pure LVOC'
    elif row['model_index'] == "1743":
        return 'Habitual'
    elif row['model_index'] == "1756":
        return 'Non-learning'
    elif row['model_index'] == "522":
        return 'SSL'
    elif row['model_index'] == "full":
        return 'Model-based'
    else:
        raise ValueError("Model class combination not found")


#
# experiment = ["v1.0", "c2.1", "c1.1"]
# experiment = ["v1.0", "c2.1", "c1.1", "high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
#               "low_variance_low_cost", "strategy_discovery"]
experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
              "low_variance_low_cost"]
# experiment = ["strategy_discovery"]

for exp in experiment:
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
        # plot_rewards(data)
        plot_clicks(data)
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        plot_rewards(data)
        # plot_clicks(data)

    # plt.ylabel("Performance", fontsize=14)
    # # text size
    # plt.rcParams.update({'font.size': 14})
    # plt.legend()
    # plt.show()
    # plt.close()
