import pandas as pd
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
import pymannkendall as mk
import statsmodels.formula.api as smf
import statsmodels.api as sm

from vars import (adaptive_pid, mod_adaptive_pid, maladaptive_pid, clicked_pid, found_optimal_twice_pid, mf_pid, \
                  hybrid_pid, habitual_pid, assign_model_names, assign_variant_names, variants, all_models,
                  habitual_not_examined_all_pid, hybrid_reinforce, mf_reinforce, not_examined_all_pid,
                  found_optimal_once_pid, found_optimal_twice_pid_not_examined_all,
                  alternative_models, mcrl_models, mb_models, optimal_pid, optimal_nonhabitual_pid)


def process_data(data, model_col, pid_col):
    data[pid_col] = data[pid_col].apply(
        lambda x: ast.literal_eval(re.sub(r'(?<=\d|\-)\s+(?=\d|\-)', ', ', x.replace('\n', ' '))) if isinstance(x,
                                                                                                                str) else x)
    data[model_col] = data[model_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data


def replace_values(r_list):
    # return [1 if x in {13.0, 14.0, 15.0, 16.0} else 0 for x in r_list]
    return [1 if x in {13.0, 14.0, 15.0} else 0 for x in r_list]


def linear_regression(data):
    # this regression analysis uses the calculated proportion of the optimal strategy instead of the booleans
    model_proportions = []
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        model_proportion = np.sum(list(model_data["model_strategy"]), axis=0) / len(model_data)
        pid_proportion = np.sum(list(model_data["pid_strategy"]), axis=0) / len(model_data)
        for trial in range(1, 121):
            model_proportions.append([model, model_proportion[trial - 1], pid_proportion[trial - 1], trial])
    data = pd.DataFrame(model_proportions, columns=["model", "model_strategy", "pid_strategy", "trial"])

    # Reshape the DataFrame
    long_df = data.melt(
        id_vars=["model", "trial"],  # Keep "model" and "trial" as identifiers
        value_vars=["model_strategy", "pid_strategy"],  # Columns to unpivot
        var_name="model_pid",  # Temporary column name
        value_name="proportion"  # Column for the proportions
    )

    # Modify the "model_pid" column to include the model name where appropriate
    long_df["model_pid"] = long_df.apply(
        lambda row: row["model"] if row["model_pid"] == "model_strategy" else "pid",
        axis=1
    )

    # Drop the "model" column as it's no longer needed
    long_df = long_df.drop(columns=["model"])

    # Filter to keep only one row per trial for "pid"
    # Because we look at the aggregated proportion of all pid, so only one entry is needed
    filtered_df = pd.concat([
        long_df[long_df["model_pid"] != "pid"],  # Keep all rows for models
        long_df[long_df["model_pid"] == "pid"].drop_duplicates(subset="trial")  # Keep one "pid" per trial
    ])

    # drop pid data if comparing between the models
    # filtered_df = filtered_df[filtered_df["model_pid"] != "pid"]

    model = smf.ols(f"proportion ~ C(model_pid, Treatment('pid')) * trial", data=filtered_df).fit()
    print(model.summary())

def plotting(model_data, model_index, criteria="model_strategy"):
    model_proportion = np.sum(list(model_data[criteria]), axis=0) / len(model_data)

    first_value = round(model_proportion[0] * 100, 1)
    last_value = round(model_proportion[-1] * 100, 1)
    print(f"Model: {model_index}, Proportion: {first_value}% to {last_value}%")
    plt.plot(list(range(1, 121)), model_proportion, label=f'{model_index}: {first_value}% to {last_value}%')

    # plt.plot(list(range(1, 121)), model_proportion, label=f'{model_index}')
    # plt.plot(list(range(1, 121)), np.sum(list(model_data['pid_strategy']), axis=0) / len(model_data), label=f'Participant')

    # add 95% CI for models
    # conf_interval = 1.96 * np.std(list(model_data[criteria])) / np.sqrt(len(list(model_data[criteria])))
    # plt.fill_between(list(range(1, 121)), model_proportion - conf_interval, model_proportion + conf_interval, alpha=0.2,
    #                  label='95% CI')

    plt.xlabel("Trial", fontsize=14)
    if criteria == "model_strategy":
        plt.ylabel("Proportion of the optimal strategy", fontsize=14)
    elif criteria == "model_rewards":
        plt.ylabel("Score", fontsize=14)
    # plt.ylim(-0.1, 1.1)
    # plt.legend(fontsize=14)
    # plt.show()
    # plt.close()


def mk_test(data, model_index, criteria="model_strategy"):
    res = mk.original_test(np.sum(list(data[criteria]), axis=0) / len(data))
    print(f"{model_index} MK test: {res}")


def linear_mixed_regression(data, criteria="pid_rewards"):
    data = data.explode(criteria)
    data["trial"] = data.groupby("pid").cumcount() + 1

    # make criteria column as int
    data[criteria] = data[criteria].astype(int)
    data["trial"] = data["trial"].astype(int)

    gamma_model_ = smf.mixedlm(formula=f"{criteria} ~ trial", data=data, groups=data["pid"]).fit()  # makes sense
    print(f"For the criteria {criteria}: ", gamma_model_.summary())


def logistic_regression(data, criteria="pid_strategy"):
    # explode both model_strategy and pid_strategy at the same time
    data = data.explode(['model_strategy', 'pid_strategy']).reset_index(drop=True)

    # data = data.explode(criteria)
    # add trial
    data["trial"] = data.groupby("pid").cumcount() + 1

    # make criteria column as int
    data[criteria] = data[criteria].astype(int)
    data["trial"] = data["trial"].astype(int)

    model = sm.GLM.from_formula(f"{criteria} ~ model_strategy * trial", data=data,
                                family=sm.families.Binomial()).fit()
    # model = sm.GLM.from_formula(f"{criteria} ~ model + model:model_strategy:trial", data=data, family=sm.families.Binomial()).fit()
    # model = sm.GLM.from_formula(f"{criteria} ~ trial", data=data, family=sm.families.Binomial()).fit()
    print(f"For the criteria {criteria}: ", model.summary())

    # use OLS
    # model = smf.ols(f"{criteria} ~  model_strategy * trial", data=data).fit()
    # print(f"For the criteria {criteria}: ", model.summary())

def load_process_data(pid_filter=None, model_type=None):
    # if name of variable is "variants", load the data from the variants folder
    model_type_name = [var_name for var_name in globals() if globals()[var_name] is model_type][0]

    if model_type_name == "variants":
        data = pd.read_csv("../../analysis/reinforce_variants_comparison/data/strategy_discovery.csv")

        # add the vanilla model
        vanilla_data = pd.read_csv(f"../../final_results/aggregated_data/strategy_discovery.csv")
        # get model_index == 3326
        vanilla_data = vanilla_data[vanilla_data["model_index"] == "3326"]
        # rename the model_index to "model"
        vanilla_data["model"] = 3326
        vanilla_data["exp"] = "strategy_discovery"
        vanilla_data = vanilla_data.drop(columns=["class", "model_index", "condition", "Unnamed: 0.1", "Unnamed: 0"])

        data = pd.concat([data, vanilla_data], ignore_index=True)
        data['model'] = data.apply(assign_variant_names, axis=1)
    else:
        data = pd.read_csv("../../final_results/aggregated_data/strategy_discovery.csv")
        data['model'] = data.apply(assign_model_names, axis=1)

    ## remove the columns and keep only pid, model, model_rewards, pid_rewards
    data = data[["pid", "model", "model_rewards", "pid_rewards"]]

    ## filter for model_types
    data = data[data['model'].isin(model_type)]

    ## Filter for certain groupd of participants
    if pid_filter:
        data = data[data['pid'].isin(pid_filter)]

    ## process data
    data = process_data(data, "model_rewards", "pid_rewards")

    ## remove row which contains nan values
    data = data.dropna()

    ## replace the values in model_rewards and pid_rewards
    data['model_strategy'] = data['model_rewards'].apply(replace_values)
    data['pid_strategy'] = data['pid_rewards'].apply(replace_values)
    return data


if __name__ == "__main__":
    ### configs
    pid = clicked_pid
    models = variants

    data = load_process_data(pid, models)

    ## pid analysis
    # linear_mixed_regression(data, criteria="pid_rewards")
    # logistic_regression(data, criteria="pid_strategy")
    linear_regression(data)

    # ### Plotting
    # plt.figure(figsize=(8, 6))
    # # Score analysis
    # for model_index in models:
    #     print("Model: ", model_index)
    #     model_data = data[data['model'] == model_index]
    #
    #     # mk_test(model_data, model_index, criteria="model_strategy")
    #     # plotting(model_data, model_index, criteria="model_strategy")
    #
    # ## Add pid proportion
    # average_score = np.sum(list(model_data['pid_strategy']), axis=0) / len(model_data)
    #
    # # first_value = round(average_score[0] * 100)
    # # last_value = round(average_score[-1] * 100)
    #
    # plt.plot(list(range(1, 121)), average_score, label=f'Participant, n={len(model_data)}', color="blue", linewidth=3)
    # # plt.plot(list(range(1, 121)), average_score, label=f'Participant, {first_value}% to {last_value}%', color="blue", linewidth=3)
    #
    # # add 95% CI for participants
    # conf_interval = 1.96 * np.std(list(model_data['pid_strategy'])) / np.sqrt(len(list(model_data['pid_strategy'])))
    # plt.fill_between(list(range(1, 121)), average_score - conf_interval, average_score + conf_interval, alpha=0.2,
    #                  label='95% CI')
    #
    #
    # plt.ylim(-0.03, 0.5)
    # plt.legend(fontsize=12, ncol=2, loc='upper left')
    # # use the name of the variable to save the plot
    # pid_name = [var_name for var_name in globals() if globals()[var_name] is pid][0]
    # model_name = [var_name for var_name in globals() if globals()[var_name] is models][0]
    #
    # # plt.savefig(f"plots/rldm/{pid_name}_{model_name}_proportion.png")
    # # plt.show()
    # # plt.close()

    # ### Proportion analysis, regression
    # for model_index in models:
    #     print("Model: ", model_index)
    #     model_data = data[data['model'] == model_index]
    #
    #     # mk_test(model_data, model_index, criteria="model_strategy")
    #     # logistic_regression(model_data, criteria="pid_strategy")
    #     linear_regression(model_data, criteria="pid_strategy")
    #     # plotting(model_data, model_index)


    # # Add pid proportion
    # proportions = np.sum(list(model_data['pid_strategy']), axis=0) / len(model_data)
    # # add number of participants in label
    # plt.plot(list(range(1, 121)), proportions, label=f'Participant, n={len(model_data)}', color="blue", linewidth=3)
    #
    # # add 95% CI for participants
    # conf_interval = 1.96 * np.std(list(model_data['pid_strategy'])) / np.sqrt(len(list(model_data['pid_strategy'])))
    # plt.fill_between(list(range(1, 121)), proportions - conf_interval, proportions + conf_interval, alpha=0.2, label='95% CI')
    #
    # plt.ylim(-0.1, 0.4)
    # plt.legend(fontsize=10, ncol=2, loc='upper left')
    # # use the name of the variable to save the plot
    # pid_name = [var_name for var_name in globals() if globals()[var_name] is pid][0]
    # model_name = [var_name for var_name in globals() if globals()[var_name] is models][0]
    #
    # plt.savefig(f"plots/cogsci2025/{pid_name}_{model_name}_strict.png")
    # plt.show()
    # plt.close()
