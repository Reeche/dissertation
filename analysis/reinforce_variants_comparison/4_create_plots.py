import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import statsmodels.formula.api as smf
from vars import learning_participants, clicking_participants, model_dict, model_names, process_clicks, process_data

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

    # for each unique model in "model" column, calculate the average of the model_mer
    for variant_type in data["model"].unique():
        data_model = data[data["model"] == variant_type]
        model_average = np.mean(data_model["model_mer"].to_list(), axis=0)

        # plot model_average, add count in label
        variant_type = model_names[variant_type]
        plt.plot(model_average, label=f"{variant_type}, {model_average[0]:.1f} to {model_average[-1]:.1f}")

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
    # plt.ylabel("Average maximum expected return")
    # plt.legend()
    # # plt.savefig(f"plots/{condition}/{model_name}_mer.png")
    # plt.show()
    # plt.close()


def plot_rewards(condition, data):
    data = data[["model", "model_rewards", "pid_rewards"]]
    # data["pid_rewards"] = data["pid_rewards"].apply(lambda x: list(ast.literal_eval(x)))
    data['pid_rewards'] = data['pid_rewards'].apply(lambda x: [int(num) for num in x[1:-1].split()])
    data['model_rewards'] = data["model_rewards"].apply(lambda x: ast.literal_eval(x))


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


    for variant_type in data["model"].unique():
        data_model = data[data["model"] == variant_type]
        model_average = np.mean(data_model[f"model_clicks"].to_list(), axis=0)

        # translate variant_type by using the dict model_names
        variant_type = model_names[variant_type]

        plt.plot(model_average, label=f"{variant_type}, {model_average[0]:.2f} to {model_average[-1]:.2f}")

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

    ### plot model_clicks and pid_clicks
    plt.plot(pid_average, label="Participant", color="blue", linewidth=3)
    plt.fill_between(x, pid_average - conf_interval, pid_average + conf_interval, color='blue', alpha=0.1,
                     label='95% CI')

    plt.ylim(0, 15)
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("Average number of clicks", fontsize=12)
    plt.legend(fontsize=10, ncol=2, loc='upper left')
    plt.savefig(f"plots/{condition}/{model_name}_clicks.png")
    # plt.show()
    plt.close()


def linear_regression(exp, data, criteria):
    """
    This function performs a linear regression on the data based on the criteria provided.
    The data format need to be reshaped into a long format with "criteria", "clicks", "trial", "model_or_pid" as columns.
    It also does pairwise comparison between the variant and the vanilla model.

    Args:
        data:
        criteria:

    Returns:

    """
    data = process_data(data, f"model_{criteria}", f"pid_{criteria}", exp)
    if criteria == "clicks":
        data["pid_clicks"] = data["pid_clicks"].apply(process_clicks)
        data["model_clicks"] = data["model_clicks"].apply(process_clicks)

    data = data[["model", "pid", f"model_{criteria}", f"pid_{criteria}"]]

    # for model in data["model"].unique():
    # todo: somehow ols over all models at the same time give insignificant results, whereas individiual model with PID give significant results
    # print(model)
    # data_filtered = data[data["model"] == model]

    # replace model name using the model_names
    data["model"] = data["model"].apply(lambda x: model_names[x])

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
# conditions = ["v1.0", "c1.1", "c2.1"]
conditions = ["c2.1"]
model_name = [3315, 3316, 3317, 3318, 3323, 3324, 3325]

for condition in conditions:
    print(condition)

    # for PRdecre
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

    # linear regression
    # linear_regression(condition, data, "clicks")
    # linear_regression(condition, data, "mer")
    # linear_regression(condition, data, "reward")

    plt.xlabel("Trial", fontsize=12)
    plt.ylim(-4, 40)
    plt.ylabel("Average most expected return", fontsize=12)
    plt.legend(fontsize=10.5, ncol=2, loc='lower left')
    plt.savefig(f"plots/{condition}/variant_mer.png")
    plt.show()
    plt.close()
