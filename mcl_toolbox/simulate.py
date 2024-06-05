import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

from mcl_toolbox.models.reinforce_models import REINFORCE
from mcl_toolbox.env.conditional_mouselab import ConditionalMouselabEnv
from mcl_toolbox.utils.learning_utils import (construct_repeated_pipeline,
                                              construct_reward_function)
from global_vars import features, structure, strategies

"""
Simulate model behaviour given chosen set of parameters
"""

possible_ground_truths = [
    [0, -1, -5, -5, -5, 1, -5, 50, -50, -1, -5, -5, -5],
    [0, -1, -5, -5, -5, 1, -5, -50, 50, -1, -5, -5, -5],
    [0, -1, -5, -5, -5, -1, -5, -5, -5, 1, -5, -50, 50],
    [0, -1, -5, -5, -5, -1, -5, -5, -5, 1, -5, 50, -50],
    [0, 1, -5, 50, -50, -1, -5, -5, -5, -1, -5, -5, -5],
    [0, 1, -5, -50, 50, -1, -5, -5, -5, -1, -5, -5, -5],
]


def cost_function(depth):
    if depth == 0:
        return 0
    if depth == 1:
        return -1
    if depth == 2:
        return -3
    if depth == 3:
        return -30


def create_env(condition, num_trials):
    reward_dist = "categorical"
    reward_structure = condition

    # reward_distributions = construct_reward_function([[-1], [-5], [-5], [-5], [1], [-5], [50], [-50], [-1], [-5], [-5], [-5]], reward_dist)

    reward_distributions = construct_reward_function(
        structure.reward_levels[reward_structure], reward_dist
    )
    repeated_pipeline = construct_repeated_pipeline(
        structure.branchings[condition], reward_distributions, num_trials
    )

    # normalized_features = get_normalized_features(structure.exp_reward_structures[condition])
    ground_truths = random.choices(possible_ground_truths, k=num_trials)
    # env = GenericMouselabEnv(num_trials, pipeline=[pipeline[0]] * num_trials)
    env = ConditionalMouselabEnv(num_trials, pipeline=[repeated_pipeline[0]] * num_trials, ground_truth=ground_truths,
                                 cost=cost_function)
    return env


def plot_score(simulation_data):
    """
    Plot the simulation data
    """
    plt.plot(simulation_data["r"][0], label="Score")
    plt.ylabel("Score")
    plt.xlabel("Trials")
    plt.legend()
    plt.show()
    # plt.savefig("score.png")
    plt.close()
    pass


def plot_clicks(simulation_data):
    """
    Plot the simulation data
    """
    # [len(sublist) - 1 for sublist in simulation_data["a"][0]]
    plt.plot([len(sublist) - 1 for sublist in simulation_data["a"][0]], label="Clicks")
    plt.ylabel("clicks")
    plt.xlabel("Trials")
    plt.legend()
    plt.show()
    # plt.savefig("clicks.png")
    plt.close()
    pass


def create_attributes(model_index):
    ###create attributes of the model
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

    if type == "hybrid":
        model_attributes_dict["features"] = features.implemented
    else:
        model_attributes_dict["features"] = features.model_free_habitual

    return model_attributes_dict, model_name


def load_attributes(parameters, condition, model_index, pid):
    data = pd.read_pickle(f"../results_sd_test2/mcrl/{condition}_priors/{pid}_likelihood_{model_index}.pkl")

    # get all prior_x in a list
    # todo: do the prior need to be sorted like 1, 2, 3 or rather like 1, 11, 12, ... 2, 21, 22, ...
    sorted_prior_keys = sorted([key for key in data[0][0].keys() if key.startswith('prior')],
                               key=lambda x: int(x.split('_')[1]))

    # Getting the values in the sorted order of keys
    sorted_prior_values = [data[0][0][key] for key in sorted_prior_keys]

    # parameters["priors"] = sorted_prior_values

    # create lsit with zeros and 1 at position 12
    prior_list = [1] * 56
    # prior_list = [random.randint(-5000, 5000) / 1000 for _ in range(56)]
    prior_list[28] = 99999 # is_leaf_and_positive_ancestor, in pkl: 'is_pos_ancestor_leaf' feature 28, starting 0
    parameters["priors"] = prior_list

    parameters["gamma"] = data[0][0]["gamma"]
    parameters["inverse_temperature"] = data[0][0]["inverse_temperature"]
    parameters["lr"] = data[0][0]["lr"]

    # if model is HR, SC, TD
    parameters["pr_weight"] = data[0][0]["pr_weight"]
    parameters["tau"] = data[0][0]["tau"]
    parameters["a"] = data[0][0]["a"]
    parameters["b"] = data[0][0]["b"]
    parameters["subjective_cost"] = data[0][0]["subjective_cost"]
    return parameters


if __name__ == "__main__":
    condition = "strategy_discovery"
    model_index = 491
    type = "hybrid"
    num_trials = 1000
    load_attribute = False
    if load_attribute:
        pid = 172

    model_attributes_dict, model_name = create_attributes(model_index)

    parameters = {}
    parameters["pr_weight"] = 1
    parameters["lik_sigma"] = 1

    env = create_env(condition, num_trials)

    for _ in range(1):  # number of simulations to run
        if model_attributes_dict["is_null"]:
            if load_attribute:
                parameters = load_attributes(parameters, condition, model_index, pid)
            else:
                parameters["priors"] = [random.randint(-5000, 5000) / 1000 for _ in range(56)]
                parameters["gamma"] = 1
                parameters["inverse_temperature"] = 1
                parameters["lr"] = 1
        else:
            if load_attribute:
                parameters = load_attributes(parameters, condition, model_index, pid)
            else:
                parameters["priors"] = [random.randint(-5000, 5000) / 1000 for _ in range(56)]
                parameters["gamma"] = random.randint(0, 1000) / 1000
                parameters["inverse_temperature"] = random.randint(0, 1000) / 1000
                parameters["lr"] = random.randint(0, 1000) / 1000

        # elif model_name == "lvoc":
        #     parameters["standard_dev"] = random.randint(0, 1000)/1000
        #     parameters["num_samples"] = random.randint(0, 10000)/1000
        #     parameters["eps"] = random.randint(0, 1000)/1000
        #     parameters["priors"] = [random.randint(-5000, 5000)/1000 for _ in range(56)]

        agent = REINFORCE(parameters, model_attributes_dict)
        simulation_data = agent.run_multiple_simulations(env=env, num_simulations=1, compute_likelihood=False,
                                                         participant=None)

        # todo: somehow the weights keep changing at the first trial
        reward = simulation_data["r"]
        # print(simulation_data["a"])
        # print("weights", agent.get_current_weights())
        plot_score(simulation_data)
        plot_clicks(simulation_data)
        # if all(value in [13, 14, 15] for sublist in reward for value in sublist):
        #     print("Model is successful. Reward: ", np.mean(reward))
        #     print("Parameters: ", parameters)
        #     print("Clicks: ", simulation_data["a"])
        #     # print("Click lengths: ", [len(sublist) for sublist in simulation_data["a"][0]])
        #     # plot_score(simulation_data)
        #     # plot_clicks(simulation_data)
        #     break
        # else:
        #     print(np.mean(reward))
