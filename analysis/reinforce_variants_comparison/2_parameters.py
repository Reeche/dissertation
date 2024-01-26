import os
import pandas as pd
import matplotlib.pyplot as plt
from vars import learning_participants, clicking_participants


"""
Create csv files with the parameters of the fitted models
"""


# model-free
def model_free(data_dir, exp):
    df = pd.DataFrame(
        columns=["exp", "pid", "model", "parameters"])

    for files in os.listdir(f"{data_dir}/{exp}_priors"):
        pid = int(files.split("_")[0])
        # model = int(files[-1].split('.')[0])
        model = int(files.split("_")[2].split('.')[0])
        if model in [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]:
            data = pd.read_pickle(f'{data_dir}/{exp}_priors/{pid}_likelihood_{model}.pkl')
            no_prior_data = {key: value for key, value in data[0][0].items() if not key.startswith('prior')}
            df.loc[len(df)] = [exp, pid, model, no_prior_data]

    df.to_csv(f"{exp}_parameters.csv")


def analyse_parameters(exp):
    df = pd.read_csv(f"{exp}_parameters.csv", index_col=0)
    if exp in ["v1.0", "c1.1", "c2.1"]:
        df = df[df["pid"].isin(clicking_participants[exp])]
    elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost",
                 "low_variance_low_cost"]:
        df = df[df["pid"].isin(learning_participants[exp])]

    df["parameters"] = df["parameters"].apply(lambda x: eval(x))
    df["subjective_cost"] = df["parameters"][0]["subjective_cost"]

    # df["subjective_cost"] = df["subjective_cost"].apply(lambda x: x["subjective_cost"])

    # plot a histogram of the subjective cost
    plt.hist(df["subjective_cost"])
    plt.show()
    print("subjective_cost", df["subjective_cost"].mean())


    return None


if __name__ == "__main__":
    ## create df
    exp_list = ['v1.0', 'c2.1', 'c1.1',
                'high_variance_high_cost',
                'high_variance_low_cost',
                'low_variance_high_cost',
                ]

    # exp_list = ["low_variance_low_cost"]

    for exp in exp_list:
        # model_free("../../results_rl_variants_8000/mcrl", exp)
        analyse_parameters(exp)
