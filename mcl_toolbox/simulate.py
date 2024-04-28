import pandas as pd
import random

from mcl_toolbox.models.lvoc_models import LVOC
from mcl_toolbox.models.reinforce_models import REINFORCE
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import (construct_repeated_pipeline,
                                              get_normalized_features,
                                              construct_reward_function)
from global_vars import features, structure, strategies
"""
Simulate model behaviour given chosen set of parameters
"""

condition = "strategy_discovery"
model_index = 491
type = "hybrid"
num_trials = 100


# create attributes of the model
model_list = pd.read_csv("models/rl_models.csv")
model_attributes = model_list.loc[model_list["index"] == model_index]
model_name = model_attributes["model"].values[0]

# replace nan value by False
model_attributes = model_attributes.where(pd.notnull(model_attributes), False)

model_attributes_dict = model_attributes.to_dict(orient='records')[0]

# add "features": features.implemented to the model_attributes_dict
model_attributes_dict["normalized_features"] = (pd.read_pickle(
    "data/normalized_values/strategy_discovery/max.pkl"), pd.read_pickle(
    "data/normalized_values/strategy_discovery/min.pkl"))
model_attributes_dict["num_priors"] = len(features.implemented)
model_attributes_dict["num_actions"] = 13
model_attributes_dict["no_term"] = False
model_attributes_dict["strategy_space"] = strategies.strategy_spaces["microscope"]


"""
    microscope = pickle_load(file_location.joinpath("data/microscope_features.pkl"))  # this is 51 features
    implemented = pickle_load(file_location.joinpath("data/implemented_features.pkl"))  # this is 56 features
    non_learning = pickle_load(file_location.joinpath("data/non_learning_features.pkl")) # 50 features
    model_free_habitual = pickle_load(file_location.joinpath("data/model_free_features.pkl")) # without MB features
"""

if type == "hybrid":
    model_attributes_dict["features"] = features.implemented
else:
    model_attributes_dict["features"] = features.model_free_habitual


## create parameters of the model
# parameters = {'prior_' + str(i): random.uniform(-2, 2) for i in range(56)}

parameters = {}

if model_name == "reinforce":
    parameters["gamma"] = random.uniform(0, 1)
    parameters["inverse_temperature"] = random.uniform(0, 1)
    parameters["lr"] = random.uniform(0, 1)
    parameters["pr_weight"] = 1
    parameters["lik_sigma"] = 1
    parameters["priors"] = [random.uniform(-2, 2) for _ in range(56)]

elif model_name == "lvoc":
    parameters["standard_dev"] = random.uniform(0, 1)
    parameters["num_samples"] = random.randint(1, 10)
    parameters["eps"] = random.uniform(0, 1)
    parameters["pr_weight"] = 1
    parameters["lik_sigma"] = 1


reward_dist = "categorical"
reward_structure = condition
reward_distributions = construct_reward_function(
    structure.reward_levels[reward_structure], reward_dist
)
repeated_pipeline = construct_repeated_pipeline(
    structure.branchings[condition], reward_distributions, num_trials
)
exp_pipelines = {condition: repeated_pipeline}

normalized_features = get_normalized_features(structure.exp_reward_structures[condition])
pipeline = exp_pipelines[condition]

env = GenericMouselabEnv(num_trials, pipeline=[pipeline[0]] * num_trials)

agent = REINFORCE(parameters, model_attributes_dict)
agent.run_multiple_simulations(env=GenericMouselabEnv, num_simulations=1, compute_likelihood=False, participant=None)