import json
import pandas as pd
import numpy as np
from mcl_toolbox.utils.learning_utils import create_dir


def split_participants_df_into_conditions(df, exp):
    """
    Split the dataframe into three csv for each condition (increasing, decreasing, constant)
    Args:
        df: dataframe

    Returns:

    """

    mf = df[df["condition"] == 0]
    stroop = df[df["condition"] == 1]
    # df_high_variance_high_click_cost = df[df["condition"] == 2]
    # df_low_variance_low_click_cost = df[df["condition"] == 3]
    #
    # df_high_variance_low_click_cost.to_csv(
    #     "../../data/human/high_variance_low_cost/participants.csv", sep=",", index=False)
    # df_low_variance_high_click_cost.to_csv(
    #     "../../data/human/low_variance_high_cost/participants.csv", sep=",", index=False)
    # df_high_variance_high_click_cost.to_csv("../../data/human/high_variance_high_cost/participants.csv", sep=",", index=False)
    # df_low_variance_low_click_cost.to_csv("../../data/human/low_variance_low_cost/participants.csv", sep=",", index=False)

    # condition = df[df["condition"] == 0]

    stroop.to_csv(f"../../data/human/stroop/participants.csv", sep=",", index=False)
    mf.to_csv(f"../../data/human/mf/participants.csv", sep=",", index=False)


def split_mouselab_df_into_conditions(df, exp):
    """
    Split the dataframe into three csv for each condition (increasing, decreasing, constant)
    Args:
        df: dataframe

    Returns:

    """
    mf = df[df["condition"] == 0]
    stroop = df[df["condition"] == 1]
    # df_high_variance_high_click_cost = df[df["condition"] == 2]
    # df_low_variance_low_click_cost = df[df["condition"] == 3]

    # condition = df[df["condition"] == 0]
    stroop.to_csv(f"../../data/human/stroop/mouselab-mdp.csv", sep=",", index=False)
    mf.to_csv(f"../../data/human/mf/mouselab-mdp.csv", sep=",", index=False)

    # df_high_variance_low_click_cost.to_csv(
    #     "../../data/human/high_variance_low_cost/mouselab-mdp.csv", sep=",", index=False)
    # df_low_variance_high_click_cost.to_csv(
    #     "../../data/human/low_variance_high_cost/mouselab-mdp.csv", sep=",", index=False)
    # df_high_variance_high_click_cost.to_csv("../../data/human/high_variance_high_cost/mouselab-mdp.csv", sep=",", index=False)
    # df_low_variance_low_click_cost.to_csv("../../data/human/low_variance_low_cost/mouselab-mdp.csv", sep=",", index=False)


experiment = "mf_stroop_full_exp"

data_full = pd.read_csv(f"data/dataclips_{experiment}.csv", sep=",")

# remove unfinished data entries
data_full["endhit"].replace("", np.nan, inplace=False)
data_full["hitid"].replace("HIT_ID", np.nan, inplace=False)
data_full.dropna(subset=["endhit"], inplace=True)
data = data_full.reset_index(drop=True)
# data.drop(index=0, inplace=True) #drops first row


data = data[["datastring"]].to_dict()  # is a df
data_value = data.get("datastring")

data_value = list(data_value.values())

# how the participants dataframe should look like containing all information
df_participants = pd.DataFrame(columns=["workerid", "condition", "bonus", "gender", "age"])
df_mouselab = pd.DataFrame()

### create temp lists to be filled to the df
# for participant csv
temp_worker_list = []
temp_condition_list = []
temp_bonus_list = []
temp_gender_list = []
temp_age_list = []

# for mouselab csv
temp_block = []
temp_trial_type = []
temp_path = []
temp_queries = []
temp_score = []
temp_all_trials_index = []
temp_state_rewards = []  # the values of ALL the nodes
temp_end_nodes = []
trial_index_list = []
pid_list = []
condition_list = []

pid_index = 1
bad_pid_list = []  # pid list of participants who failed the attention check
# for all conditions
for rows in data_value:
    row_dict = json.loads(rows)  # transform str into dict
    temp_worker_list.append(row_dict["workerId"])
    temp_condition_list.append(row_dict["condition"])
    if 'final_bonus' in row_dict["questiondata"]:
        temp_bonus_list.append(row_dict["questiondata"]["final_bonus"])
    else:
        temp_bonus_list.append(0)

    # get second to last trial to extract age and gender
    info = list(row_dict["data"])[-3]
    try:
        temp_age_list.append(info["trialdata"].get("response").get("age"))
    except:
        temp_age_list.append(None)
    try:
        temp_gender_list.append(info["trialdata"].get("response").get("gender"))
    except:
        temp_gender_list.append(None)

    # logic for trial_type: append all trials (also instructions, surveys etc, then remove them
    # Otherwise it is difficult to count when the trials appear
    temp_all_trials_index = list(range(0, len(row_dict["data"])))
    trial_index = 0

    # check if last trial is html-button-response, if yes, then the participant did not fail the attention check
    if row_dict["data"][-1].get("trialdata").get("trial_type") in ["html-button-response"]:
        # iterate through the trials
        for trial_index in temp_all_trials_index:
            # remove all none mouselab data, i.e. survey, instructions, etc
            if row_dict["data"][trial_index].get("trialdata").get("trial_type") in ["mouselab-mdp"]:
                condition_list.append(row_dict["condition"])
                trial_index_list.append(trial_index)
                pid_list.append(pid_index)
                temp_block.append(row_dict["data"][trial_index].get("trialdata").get("block"))
                temp_trial_type.append(row_dict["data"][trial_index].get("trialdata").get("trial_type"))
                temp_path.append(row_dict["data"][trial_index].get("trialdata").get("path"))
                temp_end_nodes.append(row_dict["data"][trial_index].get("trialdata").get("end_nodes"))
                temp_queries.append(row_dict["data"][trial_index].get("trialdata").get("queries"))
                temp_state_rewards.append(row_dict["data"][trial_index].get("trialdata").get("stateRewards"))
                temp_score.append(row_dict["data"][trial_index].get("trialdata").get("score"))
                trial_index += 1
        # pid_index += 1
    else:
        bad_pid_list.append(pid_index)
    pid_index += 1

print("bad pid list", bad_pid_list)

### Create mouselab csv
df_mouselab["pid"] = pid_list
df_mouselab["trial_index"] = trial_index_list
df_mouselab["condition"] = condition_list
df_mouselab["block"] = temp_block
df_mouselab["trial_type"] = temp_trial_type
df_mouselab["path"] = temp_path
df_mouselab["queries"] = temp_queries
df_mouselab["state_rewards"] = temp_state_rewards
df_mouselab["end_nodes"] = temp_end_nodes
df_mouselab["score"] = temp_score
split_mouselab_df_into_conditions(df_mouselab, experiment)
df_mouselab.to_csv(f"mouselab-{experiment}.csv", index=False, index_label="pid")

### Create participant csv
# save the information into the created df
df_participants["workerid"] = temp_worker_list
df_participants["condition"] = temp_condition_list
df_participants["bonus"] = temp_bonus_list
df_participants["gender"] = temp_gender_list
df_participants["age"] = temp_age_list
df_participants["pid"] = np.unique(pid_list).tolist()

### save participant information as csv
df_participants.index += 1
# remove bad participants
df_participants = df_participants[~df_participants.index.isin(bad_pid_list)]

split_participants_df_into_conditions(df_participants, experiment)
df_participants.to_csv(f"participants-{experiment}.csv", index=True, index_label="pid")