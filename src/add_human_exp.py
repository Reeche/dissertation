from learning_utils import pickle_load, pickle_save, get_normalized_features
exp_pipelines = pickle_load("data/exp_pipelines.pkl")
exp_reward_structures = pickle_load("data/exp_reward_structures.pkl")


new_exp_name = "IRL1"
old_exp_name = "F1" #v1.0 same as f1

exp_pipelines[new_exp_name] = exp_pipelines[old_exp_name]
#possibilities 'high_increasing', 'high_increasing', 'low_constant', 'large_increasing'
exp_reward_structures[new_exp_name] = exp_reward_structures[old_exp_name]


pickle_save(exp_pipelines, "data/exp_pipelines.pkl")
pickle_save(exp_reward_structures, "data/exp_reward_structures.pkl")