import pandas as pd
import numpy as np
import json

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


### Remember to change the number of trials!!!
# todo: extract number of trials from the data

data = pd.read_csv("data/dataclips.csv", sep=",")

# remove unfinished data entries
data["endhit"].replace("", np.nan, inplace=False)
data["hitid"].replace("HIT_ID", np.nan, inplace=False)
data.dropna(subset=["endhit"], inplace=True)
data = data.reset_index(drop=True)
# data.drop(index=0, inplace=True) #drops first row


def split_participants_df_into_conditions(df):
    """
    Split the dataframe into three csv for each condition (increasing, decreasing, constant)
    Args:
        df: dataframe

    Returns:

    """
    df_increasing = df[df["condition"] == 0]  # low cost
    df_decreasing = df[df["condition"] == 1]  # high cost
    # df_constant = df[df['condition'] == 2]

    df_increasing.to_csv(
        "../../data/human/low_cost/participants.csv", sep=",", index=False
    )
    df_decreasing.to_csv(
        "../../data/human/high_cost/participants.csv", sep=",", index=False
    )
    # df_constant.to_csv("../../data/human/c1.1/participants.csv", sep=",", index=False)


def flatten(d, sep="_"):
    """
    This function flattens json strings. It checks whether there are concatenated dicts or lists and flattens them.
    Args:
        d: data
        sep: separator to be added in between the flatted data. Example {a: {b: value}} will be flatted into {a_b: value}

    Returns: flattened OrderedDict

    """
    import collections

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):
        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    return obj


def get_queries(dict, keyword_trial, keyword_query):
    """
    Function to get information on column queries and save as dict
    Args:
        dict:

    Returns: queries data as dict

    """
    data = dict.get("data")
    queries_data = []
    for row in data:
        trialdata = row.get(keyword_trial)
        if keyword_query in trialdata:
            queries = trialdata.get(keyword_query)
            queries_data.append(queries)
    return queries_data


def format_json(df, col_name, keyword_dict, no_trials):
    """
    Flattens the csv datastring
    Args:
        df:
        col_name:
        keyword_dict:
        no_trials:

    Returns:

    """
    mouselab_dict = {}
    for index, row in df.iterrows():
        data_dict_raw = json.loads(row[col_name])
        data_dict = flatten(data_dict_raw)

        trial_dict = {}
        for trial_id in range(0, no_trials):

            all_rewards_dict = {}
            for keyword_name, keyword_len in keyword_dict.items():
                reward_key_list = []
                reward_value_list = []
                if keyword_name == "queries":
                    queries = get_queries(data_dict_raw, "trialdata", "queries")
                    for row in queries:
                        reward_value_list.append(row)
                else:
                    # create dict for all keywords
                    for key_reward in data_dict.keys():
                        if str(key_reward).find(keyword_name) != -1:
                            reward_key_list.append(key_reward)
                    for key_reward in reward_key_list:
                        reward_value_list.append(data_dict.get(key_reward))

                if keyword_name == "condition":
                    all_rewards_dict[keyword_name] = reward_value_list
                else:
                    all_rewards_dict[keyword_name] = reward_value_list[
                        (keyword_len * trial_id) : (keyword_len * (trial_id + 1))
                    ]

            trial_dict[trial_id] = all_rewards_dict
        mouselab_dict[index] = trial_dict
    return mouselab_dict


def split_mouselab_df_into_conditions(df):
    """
    Split the dataframe into three csv for each condition (increasing, decreasing, constant)
    Args:
        df: dataframe

    Returns:

    """
    df_increasing = df[df["condition"] == 0]
    df_decreasing = df[df["condition"] == 1]
    # df_constant = df[df['condition'] == 2]
    df_increasing.to_csv(
        "../../data/human/low_cost/mouselab-mdp.csv", sep=",", index=False
    )
    df_decreasing.to_csv(
        "../../data/human/high_cost/mouselab-mdp.csv", sep=",", index=False
    )
    # df_constant.to_csv("../../data/human/c1.1/mouselab-mdp.csv", sep=",", index=False)


def save_to_df(participant_dict, name_mapping):
    """
    Saves the dictionary to a csv file
    Args:
        participant_dict:
        name_mapping:

    Returns: saves a csv

    """
    dataframe_list = []
    for participant_id, trial_data in participant_dict.items():
        new_row = {}
        for trial_index, value in trial_data.items():
            new_row["pid"] = participant_id
            new_row["trial_index"] = trial_index
            for trial_type, trial_data in value.items():
                if len(trial_data) == 1:
                    new_row[trial_type] = trial_data[0]
                else:
                    new_row[trial_type] = trial_data
            row_data = new_row.copy()
            dataframe_list.append(row_data)
            # mouselab_mdp.append(new_row, ignore_index=True)
    df = pd.DataFrame(dataframe_list)

    # change the name of the dataframe
    df = df.rename(columns=name_mapping)

    df = replace_trialtype_tomouselab(df)
    split_mouselab_df_into_conditions(df)
    df.to_csv("mouselab-mdp_all.csv", sep=",", index=False)
    return df


def replace_trialtype_tomouselab(data):
    data["trial_type"] = "mouselab-mdp"
    return data


def copy_same_condition_for_all_trials():
    """
    The raw csv returns the row condition only for the first trials. This needs to be copied to all trials
    Returns:

    """
    return


# load data
data_mouselab = data[["datastring"]]

# here you can set how the columns of the csv will be named.
# here are some discrepancies between the csv output from postgres and what is required for the Computational Microscope
# left is from raw csv; right is how you want it to be
name_mapping = {
    "actionTimes": "action_time",
    "actions": "actions",
    "block": "block",
    "path": "path",
    "queries": "queries",
    "rewards": "reward",
    "rt": "rt",
    "condition": "condition",
    "bonus": "bonus",
    "score": "score",
    "simulationMode": "simulation_mode",
    "stateRewards": "state_rewards",
    "time_elapsed": "time_elapsed",
    "trial_index": "trial_index",
    "trial_time": "trialTime",
    "trial_type": "trial_type",
    "pid": "pid",
}

# here you have to enter the information you want from the csv and the length of the information
keyworddict = {
    "actionTimes": 3,
    "actions": 3,
    "block": 1,
    "path": 4,
    "queries": 1,
    "rewards": 3,
    "rt": 3,
    "condition": 1,
    "bonus": 1,
    "score": 1,
    "simulationMode": 3,
    "stateRewards": 13,
    "time_elapsed": 1,
    "trial_index": 1,
    "trial_time": 1,
    "trial_type": 1,
}

if __name__ == "__main__":
    # don't forget to change trial index
    mouselab_dict = format_json(
        data_mouselab, "datastring", keyword_dict=keyworddict, no_trials=35
    )
    df = save_to_df(mouselab_dict, name_mapping)

    # create participants csv
    # get bonus information from mouselab df and add this to the participants csv
    df["bonus"] = pd.to_numeric(df["bonus"])
    bonus_temp = df.groupby(["pid"]).sum()
    data_participants = data[["workerid", "status", "beginexp", "cond"]]
    data_participants["pid"] = bonus_temp.index
    data_participants["bonus"] = bonus_temp["bonus"]
    data_participants = data_participants.rename(columns={"cond": "condition"})

    split_participants_df_into_conditions(data_participants)
    data_participants.to_csv("participants_all.csv", index=True, index_label="pid")
