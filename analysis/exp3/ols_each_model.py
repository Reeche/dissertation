import pandas as pd
import statsmodels.formula.api as smf


def load_and_filter_data(learning_pid):
    vanilla_path = "strategy_discovery.csv"
    variant_path = "strategy_discovery_variant.csv"

    vanilla_models_data = pd.read_csv(vanilla_path)
    variant_models_data = pd.read_csv(variant_path)

    # Remove "full" and "level" models
    vanilla_models_data = vanilla_models_data[
        (vanilla_models_data["model"] != "full") & (vanilla_models_data["model"] != "level")
    ]

    # Filter for learning pid
    vanilla_models_data = vanilla_models_data[vanilla_models_data["pid"].isin(learning_pid)]
    variant_models_data = variant_models_data[variant_models_data["pid"].isin(learning_pid)]

    # Convert "model" to int (only vanilla data)
    vanilla_models_data["model"] = vanilla_models_data["model"].astype(int)

    # Concatenate dataframes
    df = pd.concat([vanilla_models_data, variant_models_data])

    # Keep relevant columns
    df = df[["pid", "model", "model_rewards", "pid_rewards"]]

    # Convert model_rewards from string to list
    df["model_rewards"] = df["model_rewards"].apply(eval)

    return df


def preprocess_model_data(df):
    # Explode model_rewards to rows
    df_model = df.explode("model_rewards")

    # Create trial_index repeating 1 to 120 for each pid*model combo
    multiplier = len(df_model) / 120
    df_model["trial_index"] = list(range(1, 121)) * int(multiplier)

    df_model = df_model.drop(columns=["pid_rewards", "pid"])
    df_model = df_model.rename(columns={"model_rewards": "score"})

    df_model["score"] = df_model["score"].astype(int)
    df_model["trial_index"] = df_model["trial_index"].astype(int)

    # Filter to first 60 trials
    df_model = df_model[df_model["trial_index"] <= 60]

    return df_model


def run_model_regressions(df_model, models):
    for model in models:
        df_ols = df_model[df_model["model"] == model]

        res = smf.ols(formula="score ~ trial_index", data=df_ols).fit()
        print(f"------------------------------------ {model} ------------------------------------")
        print(res.summary())


def run_pid_regression():
    pid_data_path = "mouselab-mdp.csv"
    pid_data = pd.read_csv(pid_data_path)
    pid_data = pid_data[["pid", "score", "trial_index"]]
    pid_data = pid_data[pid_data["trial_index"] <= 60]

    res_pid = smf.ols(formula="score ~ trial_index", data=pid_data).fit()
    print("------------------------------------ pid ------------------------------------")
    print(res_pid.summary())



if __name__ == "__main__":
    learning_pid = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 27, 28, 32, 33, 36, 37, 39, 40, 42, 43, 44,
                    46, 47, 48, 50, 51, 52, 53, 54, 55]

    models = [1743, 1756, 479, 491, 522, 480, 481, 482, 483, 484,
              485, 486, 487, 488, 489, 490, 491]

    df = load_and_filter_data(learning_pid)
    df_model = preprocess_model_data(df)
    run_model_regressions(df_model, models)
    run_pid_regression()
