import pandas as pd
import statsmodels.formula.api as smf
from vars import clicking_pid, assign_model_names
import pymannkendall as mk
import ast


def actual_score(experiment):
    dfs = []
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")

        data = data[data["pid"].isin(clicking_pid[experiment])]
        data["condition"] = experiment
        dfs.append(data)

        # formula_ = "score ~ trial_index"
        # gamma_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
        # print(gamma_model.summary())

        # linear regression
        # formula_ = "score ~ trial_index"
        # gamma_model = smf.ols(formula=formula_, data=data).fit()
        # print(gamma_model.summary())

        # Mann Kendall test
        result = mk.original_test(data["score"])
        print(result)

    # result_df = pd.concat(dfs, ignore_index=True)
    # formula_ = "score ~ trial_index:condition"
    # gamma_model = smf.mixedlm(formula=formula_, data=result_df, groups=result_df["pid"]).fit()
    # print(gamma_model.summary())

def mer(exp):
    print(exp)
    data = pd.read_csv(f"../../final_results/aggregated_data/{exp}.csv", index_col=0)
    data = data[data["pid"].isin(clicking_pid[exp])]
    # filter for model index 491; anyone would work, only need a unique list of pid
    data = data[data["model_index"] == "491"]

    data = process_data(data, "model_mer", "pid_mer")

    # explode the pid_mer and add trial numbers
    data = data.explode(['pid_mer']).reset_index(drop=False)
    data["trial_index"] = data.groupby("pid_mer").cumcount() + 1

    # check that columns "pid_mer" and "trial_index" are numeric
    data["pid_mer"] = data["pid_mer"].apply(lambda x: float(x))
    data["trial_index"] = data["trial_index"].apply(lambda x: float(x))

    formula_ = "pid_mer ~ trial_index"
    gamma_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
    print(gamma_model.summary())


def process_data(data, model_col, pid_col):
    data[pid_col] = data[pid_col].apply(lambda x: ast.literal_eval(x))
    data[model_col] = data[model_col].apply(lambda x: ast.literal_eval(x))
    return data


experiments = ["v1.0", "c2.1", "c1.1"]
for exp in experiments:
    mer(exp)