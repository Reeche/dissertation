import pandas as pd
import statsmodels.formula.api as smf

if __name__ == "__main__":
    experiments = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost",
                   "low_variance_high_cost"]
    dfs = []
    for experiment in experiments:
        data = pd.read_csv(f"../../data/human/{experiment}/mouselab-mdp.csv")
        data["condition"] = experiment
        dfs.append(data)
        # table = pd.pivot_table(data, values='score', index=['trial_index'],
        #                        columns=['pid'], aggfunc=np.sum)

        # linear mixed effect model: score ~ 1 + trial
        formula_ = "score ~ trial_index"
        gamma_model = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()
        print(gamma_model.summary())

    result_df = pd.concat(dfs, ignore_index=True)
    formula_ = "score ~ trial_index:condition"
    gamma_model = smf.mixedlm(formula=formula_, data=result_df, groups=result_df["pid"]).fit()
    print(gamma_model.summary())


