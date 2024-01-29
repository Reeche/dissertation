import matplotlib.pyplot as plt
import pandas as pd
import ast
import pymannkendall as mk
import statsmodels.formula.api as smf


def model_clicks(model, condition):
    data = pd.read_csv(f"../likelihood_vanilla_model_comparison/{condition}_1756.csv")
    # filter for model
    data = data[data['model'] == model]

    # keep only pid, model, model_clicks columns
    data = data[["pid", "model", "model_clicks", "pid_clicks"]]

    # get model clicks
    data['model_clicks'] = data['model_clicks'].apply(ast.literal_eval)
    data['pid_clicks'] = data['pid_clicks'].apply(ast.literal_eval)

    # explode model_clicks and pid_clicks
    data = data.explode(['model_clicks', 'pid_clicks']).reset_index(drop=True)

    # add trial index
    data["trial_index"] = data.groupby("pid").cumcount()

    # remove the last item in the list in model_clicks
    data["model_clicks"] = data["model_clicks"].apply(lambda x: x[:-1])
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: x[:-1])

    # get the number of clicks
    data["model_number_of_clicks"] = data["model_clicks"].apply(lambda x: len(x))
    data["pid_number_of_clicks"] = data["pid_clicks"].apply(lambda x: len(x))

    # sum number of clicks for each trial
    model_trialwise_clicks = data.groupby("trial_index")["model_number_of_clicks"].mean()
    pid_trialwise_clicks = data.groupby("trial_index")["pid_number_of_clicks"].mean()

    # linear regression on the number of clicks
    formula_ = "model_number_of_clicks ~ trial_index"
    gamma_model_ = smf.mixedlm(formula=formula_, data=data, groups=data["pid"]).fit()  # makes sense
    print(gamma_model_.summary())

    # mk test
    print(mk.original_test(model_trialwise_clicks))

    plt.plot(model_trialwise_clicks, label="model")
    plt.plot(pid_trialwise_clicks, label="pid")

    plt.legend()
    plt.show()
    plt.close()

    return None


if __name__ == "__main__":
    # conditions = ["high_variance_low_cost", "high_variance_high_cost", "low_variance_low_cost",
    #                "low_variance_high_cost"]
    conditions = ["high_variance_low_cost"]

    # models = [522, 491, 479, 1743, 1756]
    models = [1756]
    for condition in conditions:
        for model in models:
            model_clicks(model, condition)
