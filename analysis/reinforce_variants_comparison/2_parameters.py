import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from vars import learning_participants, clicking_participants, adaptive_pid, mod_adaptive_pid, maladaptive_pid, sc_adaptive, sc_maladaptive

"""
Create csv files with the parameters of the fitted models
"""


# model-free
def create_df(data_dir, selected_model: int, exp):
    df = pd.DataFrame(columns=["exp", "pid", "model", "parameters"])

    for files in os.listdir(f"{data_dir}/{exp}_priors"):
        pid = int(files.split("_")[0])
        # model = int(files[-1].split('.')[0])
        model = int(files.split("_")[2].split('.')[0])
        # if model in [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]:
        if model == selected_model:
            data = pd.read_pickle(f'{data_dir}/{exp}_priors/{pid}_likelihood_{model}.pkl')
            no_prior_data = {key: value for key, value in data[0][0].items() if not key.startswith('prior')}
            df.loc[len(df)] = [exp, pid, model, no_prior_data]

    df.to_csv(f"{exp}_{selected_model}_parameters.csv")


def merge_csvs(exp):
    # load all csv containing the name "exp"
    csvs = [file for file in os.listdir() if exp in file and file.endswith(".csv")]

    # merge all csvs into one
    df = pd.concat([pd.read_csv(csv) for csv in csvs])
    df.to_csv(f"{exp}_parameters.csv")


def analyse_parameters(exp, model, pid_group):
    df = pd.read_csv(f"{exp}_parameters.csv", index_col=0)

    ### filter for model and/or pid
    df = df[df["model"].isin(model)]
    df = df[df["pid"].isin(pid_group)]

    df["parameters"] = df["parameters"].apply(lambda x: eval(x))
    parameters_df = pd.DataFrame(list(df["parameters"]))

    # create boxplots for all parameters
    for column in parameters_df.columns:
        plt.figure()
        plt.boxplot(parameters_df[column])
        plt.title(column)
        plt.show()
        plt.close()

    # create histograms for all parameters
    for column in parameters_df.columns:
        plt.figure()
        plt.hist(parameters_df[column], bins=20)
        plt.title(column)
        plt.show()
        plt.close()

    # calculate and print the means and stds of the parameters
    print(parameters_df.mean())
    print(parameters_df.std())
    return None

def compare_parameters_between_groups(pid_groupa, pid_groupb, model):
    # compares the parameters between groups of participants
    df_ = pd.read_csv(f"{exp}_parameters.csv", index_col=0)

    ### filter for model and/or pid
    df = df_[df_["model"].isin(model)]

    # un-string it
    df["parameters"] = df["parameters"].apply(lambda x: eval(x))

    # Expand the "parameters" column into separate columns
    parameters_df = pd.json_normalize(df["parameters"])

    # Reset indices of both DataFrames to avoid index misalignment
    df = df.reset_index(drop=True)
    parameters_df = parameters_df.reset_index(drop=True)

    # Concatenate the expanded columns with the original DataFrame
    df = pd.concat([df.drop(columns=["parameters"]), parameters_df], axis=1)

    # filter for participant groups
    df_a = df[df["pid"].isin(pid_groupa)]
    df_b = df[df["pid"].isin(pid_groupb)]

    # statistically test whether the parameters are different between the groups
    for column in parameters_df.columns:
        print(column)
        # print name of pid group and mean and std of the parameter
        print("Group A")
        print(df_a[column].mean())
        print(df_a[column].std())
        print("Group B")
        print(df_b[column].mean())
        print(df_b[column].std())

        print(mannwhitneyu(df_a[column], df_b[column]))

    return None

if __name__ == "__main__":
    ## create df
    exp_list = ['v1.0', 'c2.1', 'c1.1',
                'high_variance_high_cost',
                'high_variance_low_cost',
                'low_variance_high_cost',
                ]


    models_list = [3315, 3316, 3317, 3318, 3323, 3324, 3325, 3326]

    ### Create csv
    for exp in exp_list:
        create_df(f"../../final_results/rl_hybrid_variants/hybrid", selected_model, exp)  # for variants
        create_df(f"../../final_results/hybrid", selected_model, exp) #for vanilla models

        ### Merge CSV
        merge_csvs(exp)

        ### Analyse parameters
        analyse_parameters(exp, [3315, 3316, 3323, 3324], maladaptive_pid)
        compare_parameters_between_groups(sc_adaptive, sc_maladaptive, [3315, 3316, 3323, 3324])
