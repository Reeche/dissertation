import sys

from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
# from mcl_toolbox.utils.learning_utils import pickle_load, pickle_save, get_normalized_features,\
#                             get_modified_weights, create_dir
from mcl_toolbox.computational_microscope.computational_microscope import ComputationalMicroscope
from mcl_toolbox.utils.analysis_utils import get_data


"""
Run this file to infer the averaged(?) sequences of the participants. 
Format: python3 infer_sequences.py <pid> <reward_structure> <block> 
Example: python3 infer_participant_sequences.py 1 v1.0 training
"""
#todo: here you have to enter v1.0 instead of increasing_variance. Need to make it consistent for both infer_sequence.py and infer_participant_sequences.py

def modify_clicks(click_sequence):
    modified_clicks = []
    for clicks in click_sequence:
        modified_clicks.append([int(c) for c in clicks] + [0])
    return modified_clicks

def get_participant_data(exp_num, pid, block=None):
    data = get_data(exp_num)
    clicks_data = data['mouselab-mdp']
    if block:
        clicks_data = clicks_data[(clicks_data.pid == pid) & (clicks_data.block == block)]
    else:
        clicks_data = clicks_data[clicks_data.pid == pid]
    click_sequence = [q['click']['state']['target'] for q in clicks_data.queries]
    click_sequence = modify_clicks(click_sequence)
    if 'stateRewards' in clicks_data.columns:
        envs = [[0]+sr[1:] for sr in clicks_data.stateRewards]
    elif 'state_rewards' in clicks_data.columns:
        envs = [[0]+sr[1:] for sr in clicks_data.state_rewards]
    return click_sequence, envs

def infer_strategies(click_sequences, envs, pipeline, strategy_space,
                    W, features, normalized_features):
    cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features)
    S, _, _, T = cm.infer_sequences(click_sequences, envs)
    return S, T

if __name__ == "__main__":

    #pid = int(sys.argv[1])
    # exp_num = sys.argv[2]
    #block = None
    #if len(sys.argv) > 3:
        #block = sys.argv[3]

    exp_num_list = ["v1.0", "c1.1", "c2.1_dec"]
    pid_list = [0, 3, 5, 13, 14, 18, 21, 24, 25, 31, 32, 34, 37, 42, 43]
    for i in range(0, 15):
        pid = pid_list[i]
        for j in range(0, len(exp_num_list)):
            exp_num = exp_num_list[j]
            block = "training"

            strategy_space = learning_utils.pickle_load("data/strategy_space.pkl")
            features = learning_utils.pickle_load("data/microscope_features.pkl")
            strategy_weights = learning_utils.pickle_load("data/microscope_weights.pkl")
            num_features = len(features)
            exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")

            reward_structure = "increasing_variance" #default

            exp_reward_structures = {'increasing_variance': 'high_increasing',
                                    'constant_variance': 'low_constant',
                                    'decreasing_variance': 'high_decreasing',
                                    'transfer_task': 'large_increasing'}

            reward_exps = {"increasing_variance": "v1.0",
                          "decreasing_variance": "c2.1_dec",
                          "constant_variance": "c1.1",
                          "transfer_task": "T1.1"}

            exp_reward_types = {v:k for k,v in reward_exps.items()}
            if exp_num in exp_reward_types:
                reward_structure = exp_reward_structures[exp_reward_types[exp_num]]

            #exp_num = reward_exps[reward_structure]
            # if exp_num not in exp_pipelines:
            #     raise(ValueError, "Reward structure not found.")
            pipeline = exp_pipelines["v1.0"] #default
            if exp_num in exp_pipelines:
                pipeline = exp_pipelines[exp_num]
            pipeline = [pipeline[0] for _ in range(100)]
            normalized_features = learning_utils.get_normalized_features(reward_structure)
            W = learning_utils.get_modified_weights(strategy_space, strategy_weights)
            cm = ComputationalMicroscope(pipeline, strategy_space, W, features, normalized_features=normalized_features)

            # Get clicks and envs of a particular participant
            clicks, envs = get_participant_data(exp_num, pid, block=block)
            S, T = infer_strategies(clicks, envs, pipeline, strategy_space,
                                    W, features, normalized_features)

            path = f"../results/inferred_participant_sequences/{exp_num}"
            learning_utils.create_dir(path)
            if not block:
                learning_utils.pickle_save(S, f"{path}/{pid}_strategies.pkl")
                learning_utils.pickle_save(T, f"{path}/{pid}_temperature.pkl")
            else:
                learning_utils.pickle_save(S, f"{path}/{pid}_{block}_strategies.pkl")
                learning_utils.pickle_save(T, f"{path}/{pid}_{block}_temperature.pkl")