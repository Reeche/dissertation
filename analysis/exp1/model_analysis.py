import matplotlib.pyplot as plt
import pandas as pd
import ast
import pymannkendall as mk
import statsmodels.formula.api as smf


def model_clicks(model, condition):
    data = pd.read_csv(f"../../final_results/pure_{condition}.csv")
    data = data[data['model_index'] == model][["pid", "model_index", "model_clicks", "pid_clicks"]]

    data['model_clicks'] = data['model_clicks'].apply(ast.literal_eval)
    data['pid_clicks'] = data['pid_clicks'].apply(ast.literal_eval)

    data = data.explode(['model_clicks', 'pid_clicks']).reset_index(drop=True)
    data["trial_index"] = data.groupby("pid").cumcount()

    data["model_clicks"] = data["model_clicks"].apply(lambda x: x[:-1])
    data["pid_clicks"] = data["pid_clicks"].apply(lambda x: x[:-1])

    data["model_number_of_clicks"] = data["model_clicks"].apply(len)
    data["pid_number_of_clicks"] = data["pid_clicks"].apply(len)

    model_trialwise_clicks = data.groupby("trial_index")["model_number_of_clicks"].mean()
    pid_trialwise_clicks = data.groupby("trial_index")["pid_number_of_clicks"].mean()

    formula = "model_number_of_clicks ~ trial_index"
    model_fit = smf.mixedlm(formula, data, groups=data["pid"]).fit()
    print(model_fit.summary())

    print(mk.original_test(model_trialwise_clicks))

    plt.plot(model_trialwise_clicks, label="model")
    plt.plot(pid_trialwise_clicks, label="pid")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    conditions = ["low_variance_high_cost"]
    models = [479]
    for condition in conditions:
        for model in models:
            model_clicks(model, condition)
