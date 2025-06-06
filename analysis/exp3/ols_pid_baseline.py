import pandas as pd
import statsmodels.formula.api as smf


def load_and_clean_data():
    # Load data
    vanilla_models_data = pd.read_csv("../likelihood_vanilla_model_comparison/data/strategy_discovery.csv")
    variant_models_data = pd.read_csv("../reinforce_variants_comparison/data/strategy_discovery.csv")
    pid_data = pd.read_csv("../../data/human/strategy_discovery/mouselab-mdp.csv")

    # Remove unwanted models
    vanilla_models_data = vanilla_models_data[
        (vanilla_models_data["model"] != "full") & (vanilla_models_data["model"] != "level")
        ]

    # Select relevant pid columns from pid_data
    pid_data = pid_data[["pid", "score", "trial_index"]]

    # Concatenate model datasets
    df = pd.concat([vanilla_models_data, variant_models_data])

    # Keep only relevant columns
    df = df[["pid", "model", "model_rewards", "pid_rewards"]]

    # Convert model_rewards strings to lists
    df["model_rewards"] = df["model_rewards"].apply(eval)

    # Convert pid_rewards strings to lists of ints
    df["pid_rewards"] = df["pid_rewards"].apply(lambda s: [int(num) for num in s.strip('[]').split()])

    return df, pid_data


def filter_and_prepare_data(df, pid_data):
    sig_learning_pid = [
        3, 4, 6, 7, 8, 9, 12, 15, 16, 17, 19, 21, 24, 27, 30, 32, 33,
        36, 39, 40, 42, 46, 47, 51, 52, 53, 54, 55
    ]

    # Filter for significant learning pids
    df = df[df["pid"].isin(sig_learning_pid)]

    # Explode model_rewards into rows
    df_model = df.explode("model_rewards")
    df_model["trial_index"] = df_model.groupby("pid").cumcount() + 1

    # Drop pid_rewards and pid, rename model_rewards to score
    df_model = df_model.drop(columns=["pid_rewards", "pid"])
    df_model = df_model.rename(columns={"model_rewards": "score"})

    # Add model=0 column to pid_data for baseline
    pid_data["model"] = 0

    # Combine model and pid data
    df_concat = pd.concat([df_model, pid_data])

    # Drop pid column from combined df
    df_concat = df_concat.drop(columns=["pid"], errors='ignore')

    # Convert columns to proper types
    df_concat["score"] = df_concat["score"].astype(int)
    df_concat["trial_index"] = df_concat["trial_index"].astype(int)
    df_concat["model"] = df_concat["model"].astype(str)

    # Filter to first 60 trials
    df_concat = df_concat[df_concat["trial_index"] <= 60]

    return df_concat


def run_regression(df_concat):
    model = smf.ols(formula='score ~ C(model, Treatment("0")) * trial_index', data=df_concat)
    res = model.fit()
    print(res.summary())


if __name__ == "__main__":
    df, pid_data = load_and_clean_data()
    df_concat = filter_and_prepare_data(df, pid_data)
    run_regression(df_concat)
