import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

learning_pid = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 32, 33, 36, 37, 39,
                40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 53, 54, 55]

# model data which also contains pid data
vanilla_models_data = pd.read_csv(f"../likelihood_vanilla_model_comparison/data/strategy_discovery.csv")
variant_models_data = pd.read_csv(f"../reinforce_variants_comparison/data/strategy_discovery.csv")

# remove the "full" and "level" models
vanilla_models_data = vanilla_models_data[vanilla_models_data["model"] != "full"]
vanilla_models_data = vanilla_models_data[vanilla_models_data["model"] != "level"]

# filter for learning pid
vanilla_models_data = vanilla_models_data[vanilla_models_data["pid"].isin(learning_pid)]
variant_models_data = variant_models_data[variant_models_data["pid"].isin(learning_pid)]

# convert column model to string
vanilla_models_data["model"] = vanilla_models_data["model"].astype(int)

# concat dataframes
df = pd.concat([vanilla_models_data, variant_models_data])

# keep only the columns "pid", "model", "model_rewards", "pid_rewards"
df = df[["pid", "model", "model_rewards", "pid_rewards"]]

# convert string to list
df["model_rewards"] = df["model_rewards"].apply(lambda x: eval(x))

# flatten the df, that is create a row for each item in model_rewards and pid_rewards
# only contains model rewards
df_model = df.explode("model_rewards")
# create column trial_index that ranges from 1-120 times the amount of unique pid * model combinations
multiplier = len(df_model) / 120
df_model["trial_index"] = list(range(1, 121)) * int(multiplier)

df_model = df_model.drop(columns=["pid_rewards", "pid"])
df_model = df_model.rename(columns={"model_rewards": "score"})

df_model["score"] = df_model["score"].astype(int)
df_model["trial_index"] = df_model["trial_index"].astype(int)

# filter for the first 60 trials
df_concat = df_model[df_model["trial_index"] <= 60]

# regression analysis for each model
models = [1743, 1756, 479, 491, 522, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491]
for model in models:
    # select only one model
    df_ols = df_model[df_model["model"] == model]

    # fit the model
    res = smf.ols(formula='score ~ trial_index', data=df_ols).fit()
    print("------------------------------------", model, "------------------------------------")
    print(res.summary())

### regression analysis for pid
# pid data
pid_data = pd.read_csv(f"../../data/human/strategy_discovery/mouselab-mdp.csv")
pid_data = pid_data[["pid", "score", "trial_index"]]

# filter for the first 60 trials
pid_data = pid_data[pid_data["trial_index"] <= 60]

# regression analysis for participant
res_pid = smf.ols(formula='score ~ trial_index', data=pid_data).fit()
print("------------------------------------", "pid", "------------------------------------")
print(res_pid.summary())

### winning model: 491, 479, 522
# df_ols = df_model[df_model["model"] == 491]
#
# # fit the model
# res_model = smf.ols(formula='score ~ trial_index', data=df_ols).fit()

