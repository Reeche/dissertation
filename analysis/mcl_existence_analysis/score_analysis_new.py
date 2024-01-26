import pandas as pd
import statsmodels.formula.api as smf
# from vars import clicking_pid

if __name__ == "__main__":
    experiments = ["with_click_control", "with_click_exp", "with_click_control", "no_click_exp"]
    dfs = []
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/existence_{experiment}/mouselab-mdp.csv")

        # data = data[data["pid"].isin(clicking_pid[experiment])]

        # drop all rows where "block" = "pretraining-practice"
        data = data[data["block"] != "pretraining-practice"]
        # drop the column "end_nodes"
        data = data.drop(columns=["end_nodes"])

        data["condition"] = experiment
        dfs.append(data)

        # formula_ = "score ~ trial_index"
        # gamma_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
        # print(gamma_model.summary())

    # filter for condition "with_click_exp" and block = "training"

    result_df = pd.concat(dfs, ignore_index=True)


    # for with click:
    data_with_click = result_df[(result_df["condition"] == "with_click_exp") & (result_df["block"] == "training")]
    data_no_click = result_df[(result_df["condition"] == "no_click_exp") & (result_df["block"] == "training")]

    formula_ = "score ~ trial_index:condition"
    gamma_model_with_click = smf.mixedlm(formula=formula_, data=data_with_click, groups=data_with_click["pid"]).fit()
    print(gamma_model_with_click.summary())

    gamma_model_no_click = smf.mixedlm(formula=formula_, data=data_no_click, groups=data_no_click["pid"]).fit()
    print(gamma_model_no_click.summary())



