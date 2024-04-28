import pandas as pd
import numpy as np
import ast
from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env
import matplotlib.pyplot as plt
import pymannkendall as mk
from itertools import product

exp_num = "strategy_discovery"
pid_list = get_all_pid_for_env(exp_num)
node_assumptions = ["uniform", "level", "no_assumption"]
update_rules = ["individual", "level"]

## create empty df
df = pd.DataFrame(
    columns=["pid", "node_assumption", "update_rule", "model_score", "pid_score"])
# df["pid"] = sorted(pid_list * (len(node_assumptions) + len(update_rules)))

# create all combinations of pid_list, node_assumptions, and update_rules
combinations = list(product(pid_list, node_assumptions, update_rules))

# first column of the df is first item in combinations
df["pid"] = [x[0] for x in combinations]
# second column of the df is second item in combinations
df["node_assumption"] = [x[1] for x in combinations]
# third column of the df is third item in combinations
df["update_rule"] = [x[2] for x in combinations]

### get human data
df["pid_clicks"] = "na"
df["pid_score"] = "na"
pid_info = pd.read_csv(f"../../data/human/{exp_num}/mouselab-mdp.csv")
for pid in df["pid"]:
    pid_info_temp = pid_info.loc[pid_info['pid'] == pid]
    temp_reward_list = []
    temp_click_list = []  # contains cicks of all 35 trials
    # add pid information on score and clicks
    for index, row in pid_info_temp.iterrows():
        test = row["queries"]
        test2 = ast.literal_eval(test)
        clicks = test2["click"]["state"]["target"]
        temp_click_list.append(len(clicks))
        temp_reward_list.append(row["score"])
    filter_for_pid = df.loc[df['pid'] == pid]
    for idx_, row_ in filter_for_pid.iterrows():
        df.at[idx_, 'pid_clicks'] = temp_click_list
        df.at[idx_, 'pid_score'] = temp_reward_list

### add pid score to plot
pid_score = np.array(df["pid_score"].to_list()) #take any more
average_list = pid_score.mean(axis=0)
plt.plot(range(1, len(average_list) + 1), average_list, label=f"Participant")

### Scores
for assumption in node_assumptions:
    for update_rule in update_rules:
        df_temp = df.loc[(df['node_assumption'] == assumption) & (df['update_rule'] == update_rule)]
        temp_model_score = []
        temp_model_clicks = []
        for index, row in df_temp.iterrows():
            model_data = pd.read_pickle(
                f"../../results_mb_test13/mcrl/{exp_num}_mb/{row['pid']}_likelihood_{row['node_assumption']}_{row['update_rule']}.pkl")
            temp_model_score.append(model_data["rewards"][0])
        model_score = np.array(temp_model_score)
        average_list = model_score.mean(axis=0)
        plt.plot(range(1, len(average_list) + 1), average_list, label=f"{assumption}_{update_rule}")

plt.legend()
plt.savefig("score.png")
# plt.show()
plt.close()

### Clicks
for assumption in node_assumptions:
    for update_rule in update_rules:
        df_temp = df.loc[(df['node_assumption'] == assumption) & (df['update_rule'] == update_rule)]
        temp_model_clicks = []
        for index, row in df_temp.iterrows():
            model_data = pd.read_pickle(
                f"../../results_mb_test13/mcrl/{exp_num}_mb/{row['pid']}_likelihood_{row['node_assumption']}_{row['update_rule']}.pkl")
            temp_action = []
            for action in model_data["a"][0]:
                temp_action.append(len(action))
            temp_model_clicks.append(temp_action)

        model_clicks = np.array(temp_model_clicks)
        average_list = model_clicks.mean(axis=0)
        plt.plot(range(1, len(average_list) + 1), average_list, label=f"{assumption}_{update_rule}")

        # Mann-Kendall test
        result = mk.original_test(average_list)
        print(f"Clicks: {assumption}_{update_rule}: {result}")

pid_clicks = np.array(df["pid_clicks"].to_list()) #take any more
average_list = pid_clicks.mean(axis=0)
plt.plot(range(1, len(average_list) + 1), average_list, label=f"Participant")

plt.legend()
plt.savefig("clicks.png")
# plt.show()
plt.close()
