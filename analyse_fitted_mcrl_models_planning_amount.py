import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import numpy as np
import itertools
import pymannkendall as mk
from collections import OrderedDict
from scipy.stats import sem


from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
from mcl_toolbox.analyze_sequences import analyse_sequences

"""
This file contains analysis based on fitted mcrl models.
"""


# create a dataframe of fitted models and pid
def create_dataframe_of_fitted_pid(exp_num: object, pid_list: object, number_of_trials: int, optimization_criterion: object, model_list) -> object:
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
        "BIC": []
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
                    # todo: change this to map to the json file
                    if len(data[1]) == 61: # LVOC and RF models
                        number_of_parameters = 3
                    elif len(data[1]) == 64: #hierarchical models
                        number_of_parameters = 4
                    else:
                        number_of_parameters = 99999
                    # min_loss is log-likelihood
                    # todo: check why is this plus and not 2k - 2ln(L)
                    AIC = 2 * min_loss + number_of_parameters * 2
                    results_dict["AIC"].append(AIC)
                    BIC = 2 * min_loss + number_of_parameters * np.log(number_of_trials)
                    results_dict["BIC"].append(BIC)

                    ### add reward
                    results_dict["reward_model"].append(min_loss)

    # dict to pandas dataframe
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_dict.items()]))

    AIC = False
    BIC = False

    if AIC:
        df = df.dropna(subset=["loss", "AIC"])
        # Get best model for each participant and count which one occured most often based on AIC
        df_best_model = df[df['model'].isin(model_list)]
        best_model_count = {}
        for pid in pid_list:
            data_for_pid = df_best_model[df_best_model["pid"] == pid]
            # idx_lowest = data_for_pid['AIC'].idxmin().values
            idx_lowest = data_for_pid['AIC'].idxmin()
            best_model = data_for_pid[data_for_pid.index == idx_lowest]
            best_model_name = best_model["model"].values[0]
            if best_model_name in best_model_count:
                best_model_count[best_model_name] += 1
            else:
                best_model_count[best_model_name] = 1
        print(best_model_count)

        ## sort by AIC
        df = df.sort_values(by=["AIC"])
        df["AIC"] = df["AIC"].apply(pd.to_numeric)
        grouped_df = df.groupby(["model"]).mean()
        print(exp_num)
        print("Grouped model and AIC")
        sorted_df = grouped_df.sort_values(by=["AIC"])
        print(sorted_df)

    if BIC:
        df = df.dropna(subset=["loss", "BIC"])
        ## Get best model for each participant and count which one occured most often based on BIC
        df_best_model = df[df['model'].isin(model_list)]
        best_model_count = {}
        for pid in pid_list:
            data_for_pid = df_best_model[df_best_model["pid"] == pid]
            idx_lowest = data_for_pid['BIC'].idxmin()
            best_model = data_for_pid[data_for_pid.index == idx_lowest]
            best_model_name = best_model["model"].values[0]
            if best_model_name in best_model_count:
                best_model_count[best_model_name] += 1
            else:
                best_model_count[best_model_name] = 1
        print(best_model_count)

        ## sort by BIC
        df = df.sort_values(by=["BIC"])
        df["BIC"] = df["BIC"].apply(pd.to_numeric)
        grouped_df = df.groupby(["model"]).mean()
        print(exp_num)
        print("Grouped model and BIC")
        sorted_df = grouped_df.sort_values(by=["BIC"])
        print(sorted_df)

    ## sort by loss
    # df = df.sort_values(by=['loss'])
    # df["loss"] = df["loss"].apply(pd.to_numeric)
    # grouped_df = df.groupby(["model"]).mean()
    # print(exp_num)
    # print("Grouped model and loss", grouped_df)

    ### add reward
    for model in model_list:
        average_reward = average_performance_reward(exp_num, pid_list, "likelihood", model)
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
        # plt.savefig(
        #     f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}_{plot_title}.png",
        #     bbox_inches="tight",
        # )
        # plt.show()
        # plt.close()
    return data_average


def average_performance_clicks(
        exp_num, pid_list, optimization_criterion, model_index, plot_title, plotting=True
):
    """
    Compares the number of clicks between the participant and the algorithm.

    Calculates the averaged performance for a pid list, a given optimization criterion and a model index.
    The purpose is to see the fit of a certain model and optimization criterion to a list of pid (e.g all participants from v1.0)
    Args:
        exp_num: a string "v1.0"
        pid_list: list of participant IDs
        optimization_criterion: a string
        model_index: an integer
        plotting: if true, a line plot will be created containing the average number of clicks of participant vs algo

    Returns: a dict containing averaged difference between participant and algorithm for selected model

    """
    parent_directory = Path(__file__).parents[1]
    click_info_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/click_{exp_num}_data"
    )

    all_algo_clicks = pd.DataFrame(0, index=np.arange(35), columns=pid_list)
    all_participant_clicks = pd.DataFrame(0, index=np.arange(35), columns=pid_list)

    for pid in pid_list:
        try:
            data = pickle_load(
                os.path.join(
                    click_info_directory,
                    f"{pid}_{optimization_criterion}_{model_index}.pkl",
                )
            )
        except:
            print(f"PID {pid} and model {model_index} for condition {exp_num} not found")
            continue

        # get number of clicks of algorithm (need to subtract one as one "click" action is always added by mouselab)
        all_algo_clicks[pid] = data[0] - 1

        # get number of clicks of participant
        all_participant_clicks[pid] = data[1] - 1

    # create averaged values
    participant_mean = all_participant_clicks.mean(axis=1)
    algo_mean = all_algo_clicks.mean(axis=1)

    # get standard error and standard deviation of the participants
    participant_std = all_participant_clicks.std(axis=1)
    participant_sem = all_participant_clicks.sem(axis=1)

    # calculate the difference in number of clicks
    # average difference
    average_difference = np.sum(abs(participant_mean - algo_mean))
    print(f"Average difference between number of clicks for {exp_num} and model {model_index} is {average_difference}.")

    # plot the average performance of all participants and the algorithm
    if plotting:
        plot_directory = os.path.join(parent_directory, f"mcl_toolbox/results/mcrl/plots/average/")
        create_dir(plot_directory)

        #plt.plot(algo_mean, label=model_index)
        plt.plot(algo_mean, label="Model")
        plt.plot(participant_mean, label="Participant", linewidth=3.0)
        if exp_num == "low_variance_high_cost":
            plt.axhline(y=0, color='r', linestyle='-')
        if exp_num == "low_variance_low_cost":
            plt.axhline(y=3.8, color='r', linestyle='-', label='optimal number of clicks')
        else:
            plt.axhline(y=12, color='r', linestyle='-', label='optimal number of clicks')

        # add error bars
        # plt.errorbar(x=x, y=participant_mean, yerr=participant_sem, label='error bar of participant')

        # add confidence area as shaded area
        x = range(0, 35)
        # sns.lineplot(x=x, y=participant_mean, ci=95)
        ci = 1.96 * participant_sem
        plt.fill_between(x, participant_mean-ci, participant_mean+ci, alpha=0.3, label='95% CI')

        # add labels
        plt.xlabel('Trials')
        plt.ylabel('Averaged number of clicks')
        plt.title(plot_title)

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.show()
        if exp_num in ["low_variance_high_cost", "low_variance_low_cost"]:
            plt.ylim(top=6)
        else:
            plt.ylim(top=13)
        plt.savefig(
            f"{plot_directory}/{exp_num}_{optimization_criterion}_{model_index}_{plot_title}.png",
            bbox_inches="tight",
        )
        plt.close()

    return all_participant_clicks, {model_index: average_difference}

def create_boxplots(exp_num_list, optimization_criterion, model_index):
    click_dict = {}
    for exp_num in exp_num_list:
        pid_list = get_all_pid_for_env(exp_num)
        all_participant_clicks, _ = average_performance_clicks(
            exp_num,
            pid_list,
            optimization_criterion,
            model_index=model_index,
            plot_title="",
            plotting=True,
        )
        click_dict[exp_num] = all_participant_clicks.mean(axis=1)

    click_dict["HVHC"] = click_dict.pop("high_variance_high_cost")
    click_dict["HVLC"] = click_dict.pop("high_variance_low_cost")
    click_dict["LVHC"] = click_dict.pop("low_variance_high_cost")
    click_dict["LVLC"] = click_dict.pop("low_variance_low_cost")
    fig, ax = plt.subplots()
    ax.boxplot(click_dict.values())
    ax.set_xticklabels(click_dict.keys())

    parent_directory = Path(__file__).parents[1]
    plot_directory = os.path.join(parent_directory, f"mcl_toolbox/results/mcrl/plots/boxplot")
    create_dir(plot_directory)
    plt.savefig(
        f"{plot_directory}/{optimization_criterion}_{model_index}.png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def group_adaptive_maladaptive_participant_list(exp_num, model_index):
    # Test the click sequence of each individual for trend
    parent_directory = Path(__file__).parents[1]
    click_info_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/click_{exp_num}_data")

    pid_list = get_all_pid_for_env(exp_num)
    all_participant_clicks = pd.DataFrame(0, index=np.arange(35), columns=pid_list)

    # create dataframe that contains all PID and their clicks for each trial
    for pid in pid_list:
        data = pickle_load(
            os.path.join(
                click_info_directory,
                f"{pid}_{optimization_criterion}_{model_index}.pkl",
            )
        )

        # get number of clicks of participant (need to subtract one as one "click" action is always added by mouselab)
        all_participant_clicks[pid] = data[1] - 1

    adaptive_pid = []
    maladaptive_pid = []
    other_pid = []
    # print(exp_num)
    for pid in pid_list:  # iterate through the columns
        test_results = mk.original_test(all_participant_clicks[pid])
        # print(f"Mann Kendall Test for trend for {pid}: {test_results}")

        if exp_num == "high_variance_high_cost" and test_results[0] == "increasing":
            adaptive_pid.append(pid)
        if exp_num == "high_variance_low_cost" and test_results[0] == "increasing":
            adaptive_pid.append(pid)
        if exp_num == "low_variance_high_cost" and test_results[0] == "decreasing":
            adaptive_pid.append(pid)
        if exp_num == "low_variance_low_cost" and test_results[0] == "decreasing":
            adaptive_pid.append(pid)

        if exp_num == "high_variance_high_cost" and test_results[0] == "decreasing":
            maladaptive_pid.append(pid)
        if exp_num == "high_variance_low_cost" and test_results[0] == "decreasing":
            maladaptive_pid.append(pid)
        if exp_num == "low_variance_high_cost" and test_results[0] == "increasing":
            maladaptive_pid.append(pid)
        if exp_num == "low_variance_low_cost" and test_results[0] == "increasing":
            maladaptive_pid.append(pid)

        if test_results[0] == "no trend":
            other_pid.append(pid)

    # turn results into a dict
    pid_dict = {
        "Highly adaptive participants": adaptive_pid,
        "Maladaptive participants": maladaptive_pid,
        "Mod. adaptive participants": other_pid,
    }
    return pid_dict


def create_averaged_plots_of_groups(exp_num, model_index):
    pid_dict = group_adaptive_maladaptive_participant_list(exp_num, model_index)
    for key, values in pid_dict.items():
        average_performance_clicks(exp_num, values, optimization_criterion, model_index, key, plotting=True)
    return None


# check whether the fitted parameters are significantly different across adaptive and maladaptive participants
def statistical_tests_between_groups(
        exp_num, optimization_criterion, model_index, summary=False
):
    """
    Tests whether fitted paramters are significantly different across adaptive and maladaptive participants
    Args:
        exp_num:
        optimization_criterion:
        model_index:
        summary: print out summary statistics of the fitted parameters

    Returns:

    """
    pid_dict = group_adaptive_maladaptive_participant_list(exp_num, model_index)

    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(
        parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors"
    )

    # create the df using the dictionary keys as column headers, for this, the first file in the directory is loaded

    # first_file = os.listdir(prior_directory)[6]
    # first_file = os.path.join(
    #     parent_directory, f"mcl_toolbox/results/mcrl/{exp_num}/{exp_num}_priors/{first_file}"
    # )
    # parameter_names = pickle_load(first_file)  # header are the parameters

    parameter_names = ["tau", "lr", "inverse_temperature", "gamma", "lik_sigma", "theta", "pr_weight",
                       "prior_0", "prior_1", "prior_2", "prior_3", "prior_4", "prior_5", "prior_6", "prior_7",
                       "prior_8", "prior_9", "prior_10", "prior_11", "prior_12", "prior_13", "prior_14", "prior_15",
                       "prior_16", "prior_17", "prior_18", "prior_19", "prior_20", "prior_21", "prior_22", "prior_23",
                       "prior_24", "prior_25", "prior_26", "prior_27", "prior_28", "prior_29", "prior_30", "prior_31",
                       "prior_32", "prior_33", "prior_34", "prior_35", "prior_36", "prior_37", "prior_38", "prior_39",
                       "prior_40", "prior_41", "prior_42", "prior_43", "prior_44", "prior_45", "prior_46", "prior_47",
                       "prior_48", "prior_49", "prior_50", "prior_51", "prior_52", "prior_53", "prior_54", "prior_55"]

    ## if LVOC
    # parameter_names = ["tau", "standard_dev", "num_samples", "eps", "lik_sigma", "theta", "pr_weight",
    #                    "prior_0", "prior_1", "prior_2", "prior_3", "prior_4", "prior_5", "prior_6", "prior_7",
    #                    "prior_8", "prior_9", "prior_10", "prior_11", "prior_12", "prior_13", "prior_14", "prior_15",
    #                    "prior_16", "prior_17", "prior_18", "prior_19", "prior_20", "prior_21", "prior_22", "prior_23",
    #                    "prior_24", "prior_25", "prior_26", "prior_27", "prior_28", "prior_29", "prior_30", "prior_31",
    #                    "prior_32", "prior_33", "prior_34", "prior_35", "prior_36", "prior_37", "prior_38", "prior_39",
    #                    "prior_40", "prior_41", "prior_42", "prior_43", "prior_44", "prior_45", "prior_46", "prior_47",
    #                    "prior_48", "prior_49", "prior_50", "prior_51", "prior_52", "prior_53", "prior_54", "prior_55"]

    df = pd.DataFrame(
        # index=list(parameter_names[0][0].keys()),
        index=parameter_names,
        columns=[
            "Highly adaptive participants",
            "Maladaptive participants",
            "Mod. adaptive participants",
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

                # if the participant is not in the pid_dict, i.e. a participant who used adaptive strategies in the beginning
                if plot_title is not None:
                    data = pickle_load(os.path.join(prior_directory, name))
                    for parameters, parameter_values in data[0][0].items():
                        temp_value = df.at[parameters, plot_title]
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

def create_bic_table(exp_num, pid_list, optimization_criterion, model_list):
    df = create_dataframe_of_fitted_pid(exp_num, pid_list, 35, optimization_criterion, model_list)
    df = df.drop(columns=['optimization_criterion', 'loss', 'AIC'])

    # drop not used models
    df = df[df['model'].isin(model_list)]

    df.to_csv(f"{exp_num}_bic.csv", index=False)

    df = df.reset_index()
    df = df.pivot_table(index=["pid"],columns=["model"], values=["BIC"], fill_value=0).reset_index()
    df.to_csv(f"matlab_{exp_num}_bic.csv")
    return None


if __name__ == "__main__":
    # exp_num_list = ["low_variance_high_cost"]
    exp_num_list = ['high_variance_high_cost', 'high_variance_low_cost', 'low_variance_high_cost',
                    'low_variance_low_cost']

    optimization_criterion = "number_of_clicks_likelihood"
    # model_list = ['31', '63', '95', '127', '159', '191', '607', '639', '671', '703', '735', '767',
    #               '1183', '1215', '1247', '1279', '1311', '1343', '1759', '1855',
    #               '1823', '1919', '415', '447', '479', '511', '991', '1023', '1055', '1087']

    # model_list = ['1823', '1759', '415', '31', '1919', '479', '95', '1855']  # '5038', '5134'

    model_list = ['1855']


    ###Create the plots
    for exp_num in exp_num_list:
        models_temp = {}
        pid_list = get_all_pid_for_env(exp_num)
        for model_index in model_list:
            all_participant_clicks, average_difference = average_performance_clicks(
                exp_num,
                pid_list,
                optimization_criterion,
                model_index=model_index,
                plot_title="",
                plotting=True,
            )

            models_temp.update(average_difference)

        #save multi-model plots with several models in one plot
        # plt.savefig(
        #     f"results/mcrl/plots/multi_model/{exp_num}_{optimization_criterion}_boxplots.png",
        #     bbox_inches="tight",
        # )
        # plt.show()
        # plt.close()

    ## Create boxplots for the last n trials
    # create_boxplots(exp_num_list, optimization_criterion, 1855)

    ## Compare between models
    # compare_models_df = pd.DataFrame.from_dict(models_temp, orient='index', columns=[exp_num])
    # compare_models_df = compare_models_df.sort_values()
    # models_comparison[exp_num] = compare_models_df

    # models_mean = models_comparison.mean(axis=1)
    # print(models_mean)


    ## get AIC for each group of participants
    # for exp_num in exp_num_list:
    #     pid_list = get_all_pid_for_env(exp_num)
    #     model_index = '1823'  # any is fine as we only want the participants clicks and not the models
    #     pid_dict = group_adaptive_maladaptive_participant_list(exp_num, model_index)
    #     for keys, values in pid_dict.items():
    #         print("Condition", exp_num_list, keys)
    #         df = create_dataframe_of_fitted_pid(exp_num, values, 35, optimization_criterion, model_list)


    ## Create averaged plots for each group of participants
    # for exp_num in exp_num_list:
    #     create_averaged_plots_of_groups(exp_num, model_index='1855')


    ## create a dataframe of fitted models and pid; print out the averaged loss of all models for all participants
    # for exp_num in exp_num_list:
    #     pid_list = get_all_pid_for_env(exp_num)
    #     df = create_dataframe_of_fitted_pid(exp_num, pid_list, 35, optimization_criterion)


    ##group best model by performance of participants (adaptive, maladaptive) and creates overall plot
    statistical_tests_between_groups(exp_num="low_variance_low_cost", optimization_criterion="number_of_clicks_likelihood",
                                     model_index=1855, summary=True)

    # for model_index in model_list:
        ## calculate and plot the average performance given a model for all participants
        # average_performance(exp_num, pid_list, optimization_criterion, model_index, "Averaged performance overall")

        ##calculate and plot the average performance given a model after grouping participants
        # group_adaptive_maladaptive_participant_list(exp_num_list, model_index)

    # Create and export BIC list to csv for Matlab code
    # for exp_num in exp_num_list:
    #     # exp_num = 'high_variance_high_cost'
    #     # pid_list = [0, 1, 10]'
    #     pid_list = get_all_pid_for_env(exp_num)
    #     create_bic_table(exp_num, pid_list, optimization_criterion, model_list)
