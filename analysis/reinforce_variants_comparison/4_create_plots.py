import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from vars import learning_participants

def plot_mer(data, model_name):
    # get the model_mer and pid_mer
    data = data[["model_mer", "pid_mer"]]
    data["pid_mer"] = data["pid_mer"].apply(lambda x: ast.literal_eval(x))

    data["model_mer"] = data["model_mer"].apply(lambda x: ast.literal_eval(x))

    # convert series to np array
    model = np.array(data["model_mer"].to_list())
    model_average = np.mean(model, axis=0)
    pid = np.array(data["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)


    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid) #todo:check
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(model_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue")
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1, label='95% CI')

    plt.plot(model_average, label=model_name, color="orange")

    plt.xlabel("Trial")
    # plt.ylim(0, 50)
    plt.ylabel("Average expected score")
    plt.legend()
    plt.savefig(f"plot/{exp}/{model_name}_mer.png")
    # plt.show()
    plt.close()

def plot_rewards(data, model_name):
    # get the model_mer and pid_mer
    data = data[["model_rewards", "pid_rewards"]]
    # data["pid_rewards"] = data["pid_rewards"].apply(lambda x: list(ast.literal_eval(x)))
    data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])

    model_rewards = data["model_rewards"].apply(lambda x: ast.literal_eval(x))

    # convert series to np array
    model_average = np.mean(model_rewards.to_list(), axis=0)
    pid = np.array(data["pid_rewards"].to_list())
    pid_average = np.mean(pid, axis=0)


    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid) #todo:check
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(model_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue")
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1, label='95% CI')

    plt.plot(model_average, label=model_name, color="orange")

    plt.xlabel("Trial")
    # plt.ylim(0, 50)
    plt.ylabel("Average actual score")
    plt.legend()
    plt.savefig(f"plot/{exp}/{model_name}_rewards.png")
    # plt.show()
    plt.close()

def plot_clicks(data, model_name):
    data = data[["model_clicks", "pid_clicks"]]
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: ast.literal_eval(x))

    data["model_clicks"] = data["model_clicks"].apply(lambda x: ast.literal_eval(x))

    lengths_model = []
    lengths_pid = []
    # Iterate through the DataFrame
    for index, row in data.iterrows():
        lengths_model.append([len(sublist) - 1 for sublist in row['model_clicks']])
        lengths_pid.append([len(sublist) - 1 for sublist in row['pid_clicks']])
    data["model_clicks"] = lengths_model
    data["pid_clicks"] = lengths_pid

    # convert series to np array
    model = np.array(data["model_clicks"].to_list())
    model_average = np.mean(model, axis=0)
    pid = np.array(data["pid_clicks"].to_list())
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

    plt.xlabel("Trial")
    plt.ylim(0, 10)
    plt.ylabel("Average clicks")
    plt.legend()
    # plt.savefig(f"{exp}_{model_name}_clicks.png")
    plt.show()
    plt.close()

def group_pid_according_to_bic(data):
    # group the participants into who is best explained by which model
    def sort_by_BIC(data):
        df = data.sort_values(by=["BIC"])
        average_bic = df.groupby('model')['BIC'].mean().reset_index()
        sorted_df = average_bic.sort_values(by='BIC', ascending=True)
        # the smaller BIC the better
        print(sorted_df)
        return sorted_df
    return None


model_name = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]

for model in model_name:
    exp = "strategy_discovery"
    data = pd.read_csv(f"data/{exp}.csv")

    # filter for the selected model
    data = data[data["model"] == model]

    # filter for adaptive participants
    data = data[data["pid"].isin(learning_participants[exp])]
    # data = data[data["pid"].isin([1])]

    # if exp in ["c1.1", "c2.1", "v1.0"]:
    #     plot_mer(data, model)
    #     # plot_rewards(data, model)
    # elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
    #              "low_variance_low_cost"]:

    # plot_mer(data, model)
    plot_rewards(data, model)
    # plot_clicks(data, model)


