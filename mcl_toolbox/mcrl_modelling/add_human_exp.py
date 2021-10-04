import pathlib as Path

from mcl_toolbox.utils.learning_utils import pickle_load, pickle_save

parent_folder = Path(__file__).parents[1]

exp_pipelines = pickle_load(parent_folder.joinpath("data/exp_pipelines.pkl"))
exp_reward_structures = pickle_load(
    parent_folder.joinpath("data/exp_reward_structures.pkl")
)

new_exp_name = "IRL1"
old_exp_name = "F1"  # v1.0 same as f1

exp_pipelines[new_exp_name] = exp_pipelines[old_exp_name]
# possibilities 'high_increasing', 'high_increasing', 'low_constant', 'large_increasing'
exp_reward_structures[new_exp_name] = exp_reward_structures[old_exp_name]

pickle_save(exp_pipelines, parent_folder.joinpath("/data/exp_pipelines.pkl"))
pickle_save(
    exp_reward_structures, parent_folder.joinpath("data/exp_reward_structures.pkl")
)
