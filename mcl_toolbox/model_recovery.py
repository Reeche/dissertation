import os
from pathlib import Path
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
import pandas as pd
import shutil


def create_click_sequence_csv(
    exp_num: str, optimization_criterion: str, model_index: str
):
    """
    This function goes through the created result files from the fitted mcrl models (info_{exp_num}_data)
    and extracts the click sequence.
    Args:
        exp_num: str e.g. "v1.0"
        optimization_criterion:
        model_index: str, the model index number

    Returns:

    """
    parent_directory = Path(__file__).parents[1]
    if exp_num == "c2.1":
        prior_directory = os.path.join(
            parent_directory, f"results/mcrl/{exp_num}_dec/info_{exp_num}_dec_data"
        )
    else:
        prior_directory = os.path.join(
            parent_directory, f"results/mcrl/{exp_num}/info_{exp_num}_data"
        )

    data_state_reward = pd.read_csv(
        os.path.join(
            parent_directory, f"data/original_human/{exp_num}/mouselab-mdp.csv"
        )
    )

    data_state_reward = data_state_reward[["pid", "trial_index", "state_rewards"]]

    output = {
        "pid": [],
        "trial_index": [],
        "block": [],
        "queries": [],
        "state_rewards": [],
    }

    for root, dirs, files in os.walk(prior_directory, topdown=False):
        for name in files:  # iterate through each file
            if name.endswith(f"{optimization_criterion}_{model_index}.pkl"):
                try:  # todo: not nice but it works
                    pid_ = int(name[0:3])
                except:
                    try:
                        pid_ = int(name[0:2])
                    except:
                        pid_ = int(name[0])
                # plot_title = pid_dict_reversed.get(str(pid_))
                state_rewards_for_pid = data_state_reward.loc[
                    data_state_reward["pid"] == int(pid_)
                ]
                data = pickle_load(os.path.join(prior_directory, name))

                for idx, click_sequence in enumerate(
                    data["a"][0]
                ):  # take first simulation
                    click_sequence_filtered = list(
                        filter(lambda a: a != 0, click_sequence)
                    )
                    output["pid"].append(pid_)
                    output["trial_index"].append(idx)
                    output["block"].append("training")
                    output["queries"].append(
                        f"{{'click':{{'state':{{'target': {click_sequence_filtered}}}}}}}"
                    )
                    state_rewards_ = state_rewards_for_pid.loc[
                        state_rewards_for_pid["trial_index"] == int(idx)
                    ]
                    output["state_rewards"].append(
                        state_rewards_["state_rewards"].tolist()[0]
                    )

    df_csv = pd.DataFrame.from_dict(output)
    path_save = os.path.join(parent_directory, f"data/model/{exp_num}")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    df_csv.to_csv(f"{path_save}/mouselab-mdp.csv", header=True)

    # create a copy of the participant csv in the model folder
    participants_csv = os.path.join(
        parent_directory, f"data/original_human/{exp_num}/participants.csv"
    )
    shutil.copy(participants_csv, f"{path_save}/participants.csv")


# for exp_num in ["v1.0", "c2.1", "c1.1"]:
exp_num = "c2.1"
create_click_sequence_csv(
    exp_num=exp_num, optimization_criterion="pseudo_likelihood", model_index="1853"
)
