import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import statsmodels.formula.api as smf
from vars import learning_participants, clicking_participants, model_dict, model_names

# turn off user warnings
import warnings

warnings.filterwarnings("ignore")


def plot_mer(condition, data):
    # get the model_mer and pid_mer
    data = data[["model", "model_mer", "pid_mer"]]
    data["pid_mer"] = data["pid_mer"].apply(lambda x: ast.literal_eval(x))
    data["model_mer"] = data["model_mer"].apply(lambda x: ast.literal_eval(x))

    # convert series to np array
    # model = np.array(data["model_mer"].to_list())

    linear_regression(data, "mer")

    # for each unique model in "model" column, calculate the average of the model_mer
    for variant_type in data["model"].unique():
        data_model = data[data["model"] == variant_type]
        model_average = np.mean(data_model["model_mer"].to_list(), axis=0)

        # plot model_average, add count in label
        plt.plot(model_average, label=variant_type)

    pid = np.array(data["pid_mer"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)  # todo:check
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(model_average))

    # plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    # plt.xlabel("Trial")
    # # plt.ylim(0, 50)
    # plt.ylabel("Average expected score")
    # plt.legend()
    # # plt.savefig(f"plots/{condition}/{model_name}_mer.png")
    # plt.show()
    # plt.close()


def plot_rewards(condition, data):
    data = data[["model", "model_rewards", "pid_rewards"]]
    # data["pid_rewards"] = data["pid_rewards"].apply(lambda x: list(ast.literal_eval(x)))
    data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])
    data['model_rewards'] = data["model_rewards"].apply(lambda x: ast.literal_eval(x))

    linear_regression(data, "rewards")

    # convert series to np array
    model_average = np.mean(data['model_rewards'].to_list(), axis=0)
    pid = np.array(data["pid_rewards"].to_list())
    pid_average = np.mean(pid, axis=0)

    # Calculate mean and standard error for each data point
    std_dev = np.std(pid, axis=0)
    n = len(pid)  # todo:check
    std_err = std_dev / np.sqrt(n)

    # Calculate the confidence interval
    conf_interval = 1.96 * std_err

    x = np.arange(0, len(model_average))

    ## plot model_mer and pid_mer
    plt.plot(pid_average, label="Participant", color="blue")
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    for variant_type in model_names.keys():
        data_model = data[data["model"] == variant_type]
        model_average = np.mean(data_model[f"model_rewards"].to_list(), axis=0)

        # replace model name using the model_names
        variant_type = model_names[variant_type]
        plt.plot(model_average, label=variant_type)

    # plt.xlabel("Trial")
    # # plt.ylim(0, 50)
    # plt.ylabel("Average actual score")
    # plt.legend()
    # # plt.savefig(f"plots/{condition}/{model_name}_rewards.png")
    # plt.show()
    # plt.close()


def plot_clicks(condition, data):
    data = data[["model", "model_clicks", "pid_clicks"]]
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

    linear_regression(data, "clicks")

    for variant_type in data["model"].unique():
        data_model = data[data["model"] == variant_type]
        model_average = np.mean(data_model[f"model_clicks"].to_list(), axis=0)
        plt.plot(model_average, label=variant_type)

    # convert series to np array
    # model = np.array(data["model_clicks"].to_list())
    # model_average = np.mean(model, axis=0)

    ### PID data
    # keep only unique lists in the column "pid_clicks"
    # Convert lists to tuples to use them in a set
    data['pid_clicks_tuple'] = data['pid_clicks'].apply(tuple)

    # Remove duplicates by converting the column to a set and back to a list
    unique_pid_clicks = list(set(data['pid_clicks_tuple']))

    # Convert tuples back to lists if necessary
    pid = [list(x) for x in unique_pid_clicks]

    # pid = np.array(data_unique_pid["pid_clicks"].to_list())
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

    # plt.savefig(f"plots/{condition}/{model_name}_clicks.png")
    # plt.show()
    # plt.close()


def linear_regression(data, criteria):
    """
    This function performs a linear regression on the data based on the criteria provided.
    The data format need to be reshaped into a long format with "criteria", "clicks", "trial", "model_or_pid" as columns.
    It also does pairwise comparison between the variant and the vanilla model.

    Args:
        data:
        criteria:

    Returns:

    """
    # replace model names
    data["model"] = data["model"].replace(model_names)

    # explode both model_clicks and pid_clicks at the same time
    data = data.explode([f"model_{criteria}", f"pid_{criteria}"]).reset_index(drop=True)

    # add column "trial" to the data that counts from 1-35 and repeats it for each model and pid
    times = len(data) / 35
    data["trial"] = np.tile(np.arange(0, 35), int(times))

    # make sure model_criteria and pid_criteria are integers
    data[f"model_{criteria}"] = data[f"model_{criteria}"].apply(lambda x: int(x))
    data[f"pid_{criteria}"] = data[f"pid_{criteria}"].apply(lambda x: int(x))

    # Reshape the DataFrame
    long_df = data.melt(
        id_vars=["model", "trial"],  # Keep "model" and "trial" as identifiers
        value_vars=[f"model_{criteria}", f"pid_{criteria}"],  # Columns to unpivot
        var_name="model_pid",  # Temporary column name
        value_name=f"{criteria}"  # Column for the proportions
    )

    # Modify the "model_pid" column to include the model name where appropriate
    long_df["model_pid"] = long_df.apply(
        lambda row: row["model"] if row["model_pid"] == f"model_{criteria}" else "pid",
        axis=1
    )

    # Drop the "model" column as it's no longer needed
    long_df = long_df.drop(columns=["model"])

    # model = smf.ols(f"{criteria} ~ C(model_pid, Treatment('pid')) * trial", data=long_df).fit()
    # print(model.summary())

    ## pairwise comparison
    ##filter df for vanilla model and one variant
    for variant in model_dict.keys():
        print(variant)
        filtered_df_pairwise = long_df[long_df["model_pid"].isin(["Vanilla", variant])]
        model_pairwise = smf.ols(f"{criteria} ~ C(model_pid, Treatment('Vanilla')) * trial",
                                 data=filtered_df_pairwise).fit()
        print(model_pairwise.summary())


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


# conditions = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
conditions = ["v1.0", "c1.1", "c2.1"]
model_name = [3315, 3316, 3317, 3318, 3323, 3324, 3325]

for condition in conditions:
    print(condition)

    # for PR
    data = pd.read_csv(f"data/{condition}.csv")

    if condition in ["v1.0", "c1.1", "c2.1", "strategy_discovery"]:
        data = data[data["pid"].isin(clicking_participants[condition])]
    else:
        data = data[data["pid"].isin(learning_participants[condition])]

    # use only data where model == 3318
    # data = data[data["model"].isin([3318])]

    # replace model

    # for model in model_name:
    #     data = pd.read_csv(f"data/{condition}.csv")
    #
    #     # filter for the selected model
    #     data = data[data["model"] == model]
    #
    #     # filter for adaptive participants
    #     data = data[data["pid"].isin(clicking_participants[condition])]
    #
    #     # plot_mer(condition, data, model)
    #     # plot_rewards(condition, data, model)
    plot_mer(condition, data)
    # plot_rewards(condition, data)
    # plot_clicks(condition, data) #regression analysis in this function

    # plt.xlabel("Trial", fontsize=12)
    # plt.ylim(-4, 44)
    # plt.ylabel("Average most expected reward", fontsize=12)
    # plt.legend(fontsize=11, ncol=3, loc='lower left')
    # plt.savefig(f"plots/{condition}/variant_mer.png")
    # plt.show()
    # plt.close()
