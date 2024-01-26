import os
import pandas as pd
from vars import learning_participants
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
        if model in [522, 491, 479, 1743, 1756]:
            data = pd.read_pickle(f'{data_dir}/{exp}_priors/{pid}_likelihood_{model}.pkl')
            no_prior_data = {key: value for key, value in data[0][0].items() if not key.startswith('prior')}

            df.loc[len(df)] = [exp, pid, model, no_prior_data]

    df.to_csv(f"{exp}_parameters.csv")

def model_based(data_dir, exp):
    # try:
    #     exp_ = exp.split("_")[0]
    #     df = pd.read_csv(f"{exp_}.csv", index_col=0)
    # except FileNotFoundError:
    #     print("File not created yet")
    df = pd.DataFrame(
        columns=["exp", "pid", "model", "parameters"])

    for files in os.listdir(f"{data_dir}/{exp}_mb"):
        params = {}
        pid = int(files.split("_")[0])
        data = pd.read_pickle(f'{data_dir}/{exp}_mb/{pid}_likelihood.pkl')
        params["dist_alpha"] = data["dist_alpha"]
        params["dist_beta"] = data["dist_beta"]
        params["inverse_temp"] = data["inverse_temp"]
        params["bias_inner"] = data["bias_inner"]
        params["bias_outer"] = data["bias_outer"]

        df.loc[len(df)] = [exp, pid, "mb", params]

    df.to_csv(f"parameters_{exp}.csv")


def analyse_parameters(exp):
    df = pd.read_csv(f"parameters_{exp}.csv", index_col=0)
    df = df[df["pid"].isin(learning_participants[exp])]
    # replace the parameters column with dict
    df["parameters"] = df["parameters"].apply(lambda x: eval(x))


    df["inverse_temp"] = df["parameters"].apply(lambda x: x["inverse_temp"])

    # get the mean for alpha_multiplier, dist_alpha, dist_beta, inverse_temp
    print("inverse temp", df["inverse_temp"].mean())

    return None


## create df
# exp_list = ['v1.0', 'c2.1', 'c1.1',
#             'high_variance_high_cost',
#             'high_variance_low_cost',
#             'low_variance_high_cost',
#             'low_variance_low_cost',
#             'strategy_discovery'
#             ]

exp_list = ["v1.0"]

for exp in exp_list:
    # model_based(f"../../results_mb_2000/mcrl", exp)
    model_free("../../results_mf_models_2000/mcrl", "v1.0")