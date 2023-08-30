import pandas as pd
import numpy as np
import ast
from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env
import matplotlib.pyplot as plt

exp_num = "v1.0"
pid_list = get_all_pid_for_env(exp_num)
model_list = [491, 479, "mb"] # + model_based
# model_list = ["mb"] # + model_based

# get data
df = pd.DataFrame(
    columns=["pid", "model", "model_score", "pid_score"])
df["pid"] = sorted(pid_list * len(model_list))
df["model"] = model_list * len(pid_list)



temp_model_score = []
temp_model_clicks = []
for index, row in df.iterrows():
    # load mer
    if row['model'] != "mb":
        model_data = pd.read_pickle(
            f"../../results_400_second_fit/mcrl/{exp_num}_data/{row['pid']}_{row['model']}_1.pkl")
        temp_model_score.append(model_data["r"][0])
        temp_action = []
        for action in model_data["a"][0]:
            temp_action.append(len(action))
        temp_model_clicks.append(temp_action)
    else:
        model_data = pd.read_pickle(
            f"../../results/mcrl/{exp_num}_model_based/data/{row['pid']}_likelihood.pkl")
        temp_model_score.append(model_data["rewards"][0])
        temp_action = []
        for action in model_data["a"][0]:
            temp_action.append(len(action))
        temp_model_clicks.append(temp_action)
df["model_score"] = temp_model_score
df["model_action"] = temp_model_clicks


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


### plot the mer
# for each model, get the average score
averages = {}
for model in model_list:
    model_score = np.array(df[df["model"] == model]["model_score"].to_list())
    average_list = model_score.mean(axis=0)
    averages[model] = average_list

pid_score = np.array(df[df["model"] == "mb"]["pid_score"].to_list()) #take any more
average_list = pid_score.mean(axis=0)
averages["pid"] = average_list



# Plot each dictionary
# Function to plot the values of each key as a line
def plot_dictionary_values_as_lines(dictionary):
    for key, values in dictionary.items():
        plt.plot(range(1, len(values) + 1), values, label=key)

# Plot the dictionary values as lines
# plot_dictionary_values_as_lines(averages)
#
# # Add labels, title, and legend
# plt.xlabel('Trials')
# plt.ylabel('Rewards')
# # plt.title('Values of Keys as Lines')
# plt.legend()
#
# # Show the plot
# plt.grid(True)
# plt.show()



# for each model, get the average score
averages_clicks = {}
for model in model_list:
    model_score = np.array(df[df["model"] == model]["model_action"].to_list())
    average_list = model_score.mean(axis=0)
    averages_clicks[model] = average_list

pid_score = np.array(df[df["model"] == "mb"]["pid_clicks"].to_list()) #take any more
average_list = pid_score.mean(axis=0)
averages_clicks["pid"] = average_list

# Plot the dictionary values as lines
plot_dictionary_values_as_lines(averages_clicks)

# Add labels, title, and legend
plt.xlabel('Trials')
plt.ylabel('Clicks')
# plt.title('Values of Keys as Lines')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()