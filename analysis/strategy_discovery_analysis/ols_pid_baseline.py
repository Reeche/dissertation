import pandas as pd
import statsmodels.formula.api as smf

### Regression analysis for for the actual score


# model data which also contains pid data
vanilla_models_data = pd.read_csv(f"../likelihood_vanilla_model_comparison/data/strategy_discovery.csv")
# remove the "full" and "level" models
vanilla_models_data = vanilla_models_data[vanilla_models_data["model"] != "full"]
vanilla_models_data = vanilla_models_data[vanilla_models_data["model"] != "level"]

## select only one model
# vanilla_models_data = vanilla_models_data[vanilla_models_data["model"] == 491]

variant_models_data = pd.read_csv(f"../reinforce_variants_comparison/data/strategy_discovery.csv")


# pid data
pid_data = pd.read_csv(f"../../data/human/strategy_discovery/mouselab-mdp.csv")
# keep only the columns "pid", "score", "trial_index"
pid_data = pid_data[["pid", "score", "trial_index"]]

# concat dataframes
df = pd.concat([vanilla_models_data, variant_models_data])

# keep only the columns "pid", "model", "model_rewards", "pid_rewards"
df = df[["pid", "model", "model_rewards", "pid_rewards"]]

# convert string to list
df["model_rewards"] = df["model_rewards"].apply(lambda x: eval(x))


# Apply a lambda function to convert each string representation to a list of integers
df['pid_rewards'] = df['pid_rewards'].apply(lambda s: [int(num) for num in s.strip('[]').split()])


## filter for learning pid:
learning_pid = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 32, 33, 36, 37, 39, 40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 53, 54, 55]
sig_learning_pid = [3, 4, 6, 7, 8, 9, 12, 15, 16, 17, 19, 21, 24, 27, 30, 32, 33, 36, 39, 40, 42, 46, 47, 51, 52, 53, 54, 55]

df = df[df["pid"].isin(sig_learning_pid)]


# flatten the df, that is create a row for each item in model_rewards and pid_rewards
# only contains model rewards
df_model = df.explode("model_rewards")
df_model["trial_index"] = df_model.groupby("pid").cumcount() + 1
df_model = df_model.drop(columns=["pid_rewards", "pid"])
df_model = df_model.rename(columns={"model_rewards": "score"})

# replace pid_data "pid" with value 0
pid_data["model"] = 0

# concat df_model and pid_data
df_concat = pd.concat([df_model, pid_data])

# drop "pid" column
df_concat = df_concat.drop(columns=["pid"])

# convert "score" and "trial_index" columns to integer
df_concat["score"] = df_concat["score"].astype(int)
df_concat["trial_index"] = df_concat["trial_index"].astype(int)

# convert column model to string
df_concat["model"] = df_concat["model"].astype(str)

# filter for the first 60 trials
df_concat = df_concat[df_concat["trial_index"] <= 60]

# filter for learning


# regression analysis with model 0 as baseline
model = smf.ols(formula='score ~ C(model, Treatment("0"))*trial_index', data=df_concat)
print(model.fit().summary())
