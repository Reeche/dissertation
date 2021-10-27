import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import numpy as np
import itertools

from mcl_toolbox.utils.utils import get_all_pid_for_env
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
from mcl_toolbox.analyze_sequences import analyse_sequences

"""
This file contains analysis based on fitted mcrl models.
"""


# create a dataframe of fitted models and pid
def create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion):
    """
    This function loops through all pickles and creates a dataframe with corresponding loss of each model.

    Args:
        exp_num: experiment number, e.g. c2.1_dec
        pid_list: list of pid
        optimization_criterion: e.g. "pseudo_likelihood"

    Returns: A dataframe with the columns ["optimization_criterion", "model", "loss"] and index is the pid

    """
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors"
    )
    # todo: need to find a better way to create this df
    # df = pd.DataFrame(columns=["pid", "optimization_criterion", "model", "loss", "AIC"])
    results_dict = {
        "pid": [],
        "model": [],
        "optimization_criterion": [],
        "loss": [],
        "AIC": [],
    }

    for root, dirs, files in os.walk(prior_directory, topdown=False):
        for name in files:
            for pid in pid_list:
                if name.startswith(f"{pid}_{optimization_criterion}") and name.endswith(
                        ".pkl"
                ):
                    results_dict["pid"].append(pid)
                    data = pickle_load(os.path.join(prior_directory, name))
                    if name[-8:-4].startswith("_"):
                        results_dict["model"].append(name[-7:-4])
                        # df.loc[pid]["model"] = name[-7:-4]  # get the last 3 characters, which are the model name
                    elif name[-9:-4].startswith("_"):
                        results_dict["model"].append(name[-8:-4])
                        # df.loc[pid]["model"] = name[-8:-4]  # get the last 4 characters, which are the model name
                    else:
                        results_dict["model"].append(name[-6:-4])  # get the last 2 characters, which are the model name

                    results_dict["optimization_criterion"].append(
                        optimization_criterion
                    )
                    # df.loc[pid]["optimization_criterion"] = optimization_criterion

                    losses = [trial["result"]["loss"] for trial in data[0][1]]
                    min_loss = min(np.absolute(losses))
                    results_dict["loss"].append(min_loss)
                    # df.loc[pid]["loss"] = min_loss

                    # Calculate the AIC
                    number_of_parameters = len(data[1])
                    AIC = 2 * min_loss + 2 * number_of_parameters
                    # df.loc[pid]["AIC"] = AIC
                    results_dict["AIC"].append(AIC)

    # dict to pandas dataframe
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_dict.items()]))
    df = df.dropna(subset=["loss", "AIC"])

    ## Get best model for each participant and count which one occured most often
    # df_best_model = df[df["model"].isin(["1853", "1757", "5134", "2022"])]
    # best_model_count = {}
    # for pid in pid_list:
    #     data_for_pid = df_best_model[df_best_model["pid"] == pid]
    #     idx_lowest = data_for_pid[['AIC']].idxmin().values
    #     best_model = data_for_pid[data_for_pid.index == idx_lowest[0]]
    #     best_model_name = best_model["model"].values[0]
    #     if best_model_name in best_model_count:
    #         best_model_count[best_model_name] += 1
    #     else:
    #         best_model_count[best_model_name] = 1
    # print(best_model_count)

    ## sort by loss
    # df = df.sort_values(by=['loss'])
    # df["loss"] = df["loss"].apply(pd.to_numeric)
    # grouped_df = df.groupby(["model"]).mean()
    # print(exp_num)
    # print("Grouped model and loss", grouped_df)

    ## sort by AIC
    df = df.sort_values(by=["AIC"])
    df["AIC"] = df["AIC"].apply(pd.to_numeric)
    grouped_df = df.groupby(["model"]).mean()
    print(exp_num)
    print("Grouped model and AIC")
    sorted_df = grouped_df.sort_values(by=["AIC"])
    print(sorted_df)

    return df


def average_performance_reward(
        exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True
):
    """
    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num: a string "v1.0"
        pid_list: list of participant IDs
        optimization_criterion: a string
        model_index: an integer
        plotting: if true, plots are created

    Returns: average performance of model and participant

    """
    parent_directory = Path(__file__).parents[1]
    reward_info_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/reward_{exp_num}_data"
    )

    data_temp = pickle_load(
        os.path.join(
            reward_info_directory,
            f"{pid_list[0]}_{optimization_criterion}_{model_index}.pkl",
        )
    )
    number_of_participants = len(pid_list)

    for pid in pid_list:
        data = pickle_load(
            os.path.join(
                reward_info_directory,
                f"{pid}_{optimization_criterion}_{model_index}.pkl",
            )
        )
        data_temp["Reward"] = data_temp["Reward"].add(data["Reward"])

    # create averaged values
    data_average = data_temp
    data_average["Reward"] = data_temp["Reward"] / number_of_participants

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"mcl_toolbox/results/mcrl/plots/average/")
        create_dir(plot_directory)
        # plt.ylim(-30, 30)
        plt.title(plot_title)
        ax = sns.lineplot(
            x="Number of trials", y="Reward", hue="Type", data=data_average
        )
        plt.savefig(
            f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}_{plot_title}.png",
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
    return data_average


def average_performance_clicks(
        exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True
):
    """
    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num: a string "v1.0"
        pid_list: list of participant IDs
        optimization_criterion: a string
        model_index: an integer
        plotting: if true, plots are created

    Returns: average performance of model and participant

    """
    parent_directory = Path(__file__).parents[1]
    click_info_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/click_{exp_num}_data"
    )

    data_aggregated = pd.DataFrame(0, index=np.arange(35), columns=["algorithm", "participant"])
    number_of_participants = len(pid_list)

    for pid in pid_list:
        data = pickle_load(
            os.path.join(
                click_info_directory,
                f"{pid}_{optimization_criterion}_{model_index}.pkl",
            )
        )
        # get number of clicks of algorithm
        data_aggregated["algorithm"] = data_aggregated["algorithm"] + data[0]

        # get number of clicks of participant
        data_aggregated["participant"] = data_aggregated["participant"] + data[1]

    # create averaged values
    data_aggregated["algorithm"] = data_aggregated["algorithm"] / number_of_participants
    data_aggregated["participant"] = data_aggregated["participant"] / number_of_participants

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"mcl_toolbox/results/mcrl/plots/average/")
        create_dir(plot_directory)

        plt.plot(data_aggregated["algorithm"], label="Algorithm")
        plt.plot(data_aggregated["participant"], label="Participant")
        plt.xlabel('Trials')
        plt.ylabel('Averaged number of clicks')
        plt.legend()
        #plt.show()
        plt.savefig(
            f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}_{plot_title}.png",
            bbox_inches="tight",
        )
        plt.close()
    return data_aggregated


def get_adaptive_maladaptive_participant_list(exp_num):
    """

    Args:
        exp_num: a string, e.g. "v1.0"

    Returns: a dictionary {str: list}

    """
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        adaptive_participants,
        maladaptive_participants,
        other_participants,
        improved_participants,
    ) = analyse_sequences(
        exp_num,
        num_trials=35,
        block="training",
        pids=None,
        create_plot=False,
        number_of_top_worst_strategies=5,
    )
    pid_dict = {
        "Adaptive strategies participants": improved_participants,
        "Maladaptive strategies participants": maladaptive_participants,
        "Other strategies participants": other_participants,
    }
    return pid_dict


def group_by_adaptive_malapdaptive_participants(
        exp_num, optimization_criterion, model_index=None, plotting=True
):
    """
    This function groups participants into adaptive/maladaptive/others (if they have used adaptive strategies in their last trial,
    they are adaptive) and creates plots based on the grouping

    Args:
        exp_num: e.g. "v1.0"
        model_index: the model index, e.g. 861

    Returns: nothing

    """
    pid_dict = get_adaptive_maladaptive_participant_list(exp_num)

    for plot_title, pid_list in pid_dict.items():
        if plotting:
            average_performance_reward(
                exp_num,
                pid_list,
                optimization_criterion,
                model_index,
                plot_title,
                plotting=True,
            )
        else:
            # get the best model for each participant group
            print(f"Grouped for {plot_title} participants")
            create_dataframe_of_fitted_pid(exp_num, pid_list, optimization_criterion)
            # print(f"The best model with lowest AIC for {exp_num}, {plot_title} is {best_model}")

    return None


# check whether the fitted parameters are significantly different across adaptive and maladaptive participants
def statistical_tests_between_groups(
        exp_num, optimization_criterion, model_index, summary=False
):
    pid_dict = get_adaptive_maladaptive_participant_list(exp_num)

    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors"
    )

    # create the df using the dictionary keys as column headers, for this, the first file in the directory is loaded

    first_file = os.listdir(prior_directory)[10]
    first_file = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors/{first_file}"
    )
    parameter_names = pickle_load(first_file)  # header are the parameters

    df = pd.DataFrame(
        index=list(parameter_names[0][0].keys()),
        columns=[
            "Adaptive strategies participants",
            "Maladaptive strategies participants",
            "Other strategies participants",
        ],
    )
    pid_dict_reversed = {}
    for k, v in pid_dict.items():
        for v_ in v:
            pid_dict_reversed[str(v_)] = k

    for root, dirs, files in os.walk(prior_directory, topdown=False):
        for name in files:  # iterate through each file
            if name.endswith(f"{optimization_criterion}_{model_index}.pkl"):
                try:  # todo: not nice but it works
                    pid_ = int(name[0:3])
                except:
                    try:
                        pid_ = int(name[0:2])
                    except:
                        pid_ = int(name[0])
                plot_title = pid_dict_reversed.get(str(pid_))
                if (
                        plot_title is not None
                ):  # if the participant is not in the pid_dict, i.e. a participant who used adaptive strategies in the beginning
                    data = pickle_load(os.path.join(prior_directory, name))
                    for parameters, parameter_values in data[0][0].items():
                        temp_value = df.loc[parameters, plot_title]
                        if type(temp_value) == float or temp_value is None:
                            df.at[parameters, plot_title] = [parameter_values]
                        else:
                            temp_value.append(parameter_values)
                            df.at[parameters, plot_title] = temp_value
                else:
                    continue

    # summary statistic of the parameters
    if summary:
        for index, row in df.iterrows():  # iterate through each row
            row_names = row.index
            i = 0
            for (
                    columns
            ) in (
                    row
            ):  # in each row, take out one column, i.e. adaptive/maladaptive/other
                print(
                    f"------------------Summary statitics of {row_names[i]}------------------"
                )
                parameter_mean = np.mean(np.exp(columns))
                print(f"Mean of parameter {index} is: {parameter_mean}")
                parameter_sd = np.std(np.exp(columns))
                print(f"Standard deviation of parameter {index} is: {parameter_sd}")
                print("Q1 quantile of arr : ", np.quantile(np.exp(columns), 0.25))
                print("Q3 quantile of arr : ", np.quantile(np.exp(columns), 0.75))

                # test for normality
                # try:
                #     k2, p = stats.normaltest(np.exp(columns))  # first time without exp
                #     if p < 0.05:  # null hypothesis: x comes from a normal distribution
                #         print(f"{row_names[i]} is not normally distributed")
                #         # print(f"The p-value for normality for {row.index} is {p}")
                #     else:
                #         print(f"{row_names[i]} is normally distributed")
                # except:
                #     continue

                i += 1

    # t-test or Wilcoxon rank rum test of parameters of unequal variance
    for index, row in df.iterrows():
        row_names = row.index
        i = 0
        for column_a in row:
            j = 0
            for column_b in row:
                if column_a != column_b:
                    # test_statistic, p = stats.ttest_ind(np.exp(column_a), np.exp(column_b), equal_var=False)
                    try:
                        test_statistic, p = stats.ranksums(
                            np.exp(column_a), np.exp(column_b)
                        )
                        if p < 0.05:  # print out only the significant ones
                            print(
                                f"---------------Testing parameter {index} between {row_names[i]} vs {row_names[j]}---------------"
                            )
                            print(f"Test statistic: {test_statistic}")
                            print(f"p-value: {p}")
                    except:
                        print(
                            f"Skipped {index} between {row_names[i]} vs {row_names[j]}"
                        )
                        continue
                j += 1
            i += 1


def statistical_test_between_envs(exp_num_list, model_index):
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num_list[0]}/{exp_num_list[0]}_priors"
    )

    first_file = os.listdir(prior_directory)[10]
    first_file = os.path.join(
        parent_directory,
        f"mcl_toolbox/results/mcrl/{exp_num_list[0]}/{exp_num_list[0]}_priors/{first_file}",
    )
    parameter_names = pickle_load(first_file)  # header are the parameters

    df = pd.DataFrame(
        index=list(parameter_names[0][0].keys()), columns=["v1.0", "c2.1_dec", "c1.1"]
    )

    for exp_num in exp_num_list:
        prior_directory = os.path.join(
            parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors"
        )
        for root, dirs, files in os.walk(prior_directory, topdown=False):
            for name in files:  # iterate through each file
                if name.endswith(f"{model_index}.pkl"):
                    data = pickle_load(os.path.join(prior_directory, name))
                    for parameters, parameter_values in data[0][0].items():
                        temp_value = df.loc[parameters, exp_num]
                        if type(temp_value) == float or temp_value is None:
                            df.at[parameters, exp_num] = [parameter_values]
                        else:
                            temp_value.append(parameter_values)
                            df.at[parameters, exp_num] = temp_value
    # t-test or Wilcoxon rank rum test of parameters of unequal variance
    for index, row in df.iterrows():
        column_names = row.index
        for pair in itertools.combinations(column_names, 2):
            try:
                test_statistic, p = stats.ranksums(
                    np.exp(row[pair[0]]), np.exp(row[pair[1]])
                )
                if p < 0.05:  # print out only the significant ones
                    print(
                        f"---------------Testing parameter {index} between {pair[0]} vs {pair[1]}---------------"
                    )
                    print(f"Test statistic: {test_statistic}")
                    print(f"p-value: {p}")
            except:
                print(f"Skipped {index} between {pair[0]} vs {pair[1]}")
                continue


if __name__ == "__main__":
    # exp_num = sys.argv[1]  # e.g. c2.1_dec
    # optimization_criterion = sys.argv[2]

    # exp_num_list = ["v1.0", "c2.1_dec", "c1.1"]
    exp_num_list = ['high_variance_high_cost', 'high_variance_low_cost', 'low_variance_high_cost',
                    'low_variance_low_cost']

    optimization_criterion = "clicks_overlap"
    # model_list = ['31', '63', '95', '127', '159', '191', '607', '639', '671', '703', '735', '767',
    #           '1183', '1215', '1247', '1279', '1311', '1343', '1759', '1855']
    model_list = ['1823', '1919', '415', '447', '479', '511', '991', '1023', '1055', '1087']

    # statistical_test_between_envs(exp_num_list, model_index=1918)
    # Run t-test and statistical summary
    # for exp_num in exp_num_list:
    #     for model_index in model_list:
    #         pid_list = get_all_pid_for_env(exp_num)
    #         average_performance(
    #             exp_num,
    #             pid_list,
    #             optimization_criterion,
    #             model_index=model_index,
    #             plot_title="",
    #             plotting=True,
    #         )
    average_performance_clicks(
        'low_variance_low_cost',
        ['3', '5', '6'],
        optimization_criterion,
        model_index=159,
        plot_title="",
        plotting=True,
    )
    # create a dataframe of fitted models and pid; print out the averaged loss of all models for all participants
    df = create_dataframe_of_fitted_pid('low_variance_low_cost', '3', optimization_criterion)

    # # group best model by performance of participants (adaptive, maladaptive) and creates overall plot
    # group_by_adaptive_malapdaptive_participants(exp_num, optimization_criterion, model_index=1853, plotting=False) #if plotting, then it needs model_index
    #
    # statistical_tests_between_groups(exp_num=exp_num, optimization_criterion="pseudo_likelihood", model_index=1853,
    #                                  summary=True)
    #
    #
    # for model_index in model_list:
    #     # calculate and plot the average performance given a model for all participants
    #     average_performance(exp_num, pid_list, optimization_criterion, model_index, "Averaged performance overall")
    #
    #     # calculate and plot the average performance given a model after grouping participants
    #     group_by_adaptive_malapdaptive_participants(exp_num, optimization_criterion, model_index, plotting=True)