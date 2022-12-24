import ast
import sys
import time
from pathlib import Path

from mcl_toolbox.utils.learning_utils import create_dir
from mcl_toolbox.utils.model_utils import ModelFitter
import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt

import pickle

import numpy as np

exp_name = "scarcity_scarce"
model_index = 6527
model_index_null = 6526
optimization_criterion = "likelihood"

# other_params = {"plotting": True}
other_params_1 = {}
other_params_2 = {}
other_params_3 = {}
other_params_4 = {}
other_params_5 = {}
other_params_6 = {}
other_params_7 = {}
other_params_8 = {}
other_params_9 = {}

click_cost = 0.25 if exp_name == "scarcity_scarce" else 1
number_of_trials = 120 if exp_name == "scarcity_scarce" else 30

pid = "631a24dfca851aac1863e4a4" if exp_name == "scarcity_scarce" else "60fd5adad03767cff6dddda8"
# 631a24dfca851aac1863e4a4 - example scarce
# 60fd5adad03767cff6dddda8 - example control

exp_attributes = {
    "exclude_trials": None,  # Trials to be excluded
    "block": None,  # Block of the experiment
    "experiment": None,
    # Experiment object can be passed directly with pipeline and normalized features attached
    "click_cost": click_cost,
}
other_params_1["exp_attributes"] = exp_attributes.copy()
other_params_2["exp_attributes"] = exp_attributes.copy()
other_params_3["exp_attributes"] = exp_attributes.copy()
other_params_4["exp_attributes"] = exp_attributes.copy()
other_params_5["exp_attributes"] = exp_attributes.copy()
other_params_6["exp_attributes"] = exp_attributes.copy()
other_params_7["exp_attributes"] = exp_attributes.copy()
other_params_8["exp_attributes"] = exp_attributes.copy()
other_params_9["exp_attributes"] = exp_attributes.copy()

optimization_params = {
    "optimizer": "hyperopt",
    "num_simulations": 1,  # likelihood doesn't need more than 1
    "max_evals": 2,  # 400 - number of param updates
    "compute_all_likelihoods": False
}

other_params_1["optimization_params"] = optimization_params.copy()
other_params_2["optimization_params"] = optimization_params.copy()
other_params_3["optimization_params"] = optimization_params.copy()
other_params_4["optimization_params"] = optimization_params.copy()
other_params_5["optimization_params"] = optimization_params.copy()
other_params_6["optimization_params"] = optimization_params.copy()
other_params_7["optimization_params"] = optimization_params.copy()
other_params_8["optimization_params"] = optimization_params.copy()
other_params_9["optimization_params"] = optimization_params.copy()

other_params_1["optimization_params"]["learn_from_actions"] = 1
other_params_2["optimization_params"]["learn_from_actions"] = 1
other_params_3["optimization_params"]["learn_from_actions"] = 0.015
other_params_5["optimization_params"]["learn_from_actions"] = 1
other_params_7["optimization_params"]["learn_from_actions"] = 1
other_params_8["optimization_params"]["learn_from_actions"] = 0
other_params_9["optimization_params"]["learn_from_actions"] = 0

other_params_1["optimization_params"]["max_integration_degree"] = 7
other_params_2["optimization_params"]["max_integration_degree"] = 7
other_params_4["optimization_params"]["max_integration_degree"] = 7
other_params_5["optimization_params"]["max_integration_degree"] = 7
other_params_7["optimization_params"]["max_integration_degree"] = 7
other_params_8["optimization_params"]["max_integration_degree"] = 7
other_params_9["optimization_params"]["max_integration_degree"] = 6

other_params_2["optimization_params"]["learn_from_unrewarded"] = True

other_params_1["optimization_params"]["ignore_reward"] = True

other_params_1["optimization_params"]["learn_from_PER"] = 1
other_params_2["optimization_params"]["learn_from_PER"] = 2
other_params_4["optimization_params"]["learn_from_PER"] = 2
other_params_5["optimization_params"]["learn_from_PER"] = 1

other_params_7["optimization_params"]["learn_from_MER"] = True


mf = ModelFitter(
    exp_name,
    exp_attributes=exp_attributes,
    data_path=None,
    number_of_trials=number_of_trials,
)

optimizer_1 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_1["optimization_params"])
optimizer_1.optimizer = "hyperopt"
optimizer_1.num_simulations = 1

optimizer_1_n = mf.construct_optimizer(model_index_null, pid, optimization_criterion, other_params_1["optimization_params"])
optimizer_1_n.optimizer = "hyperopt"
optimizer_1_n.num_simulations = 1

# print(other_params_2)
optimizer_2 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_2["optimization_params"])
optimizer_2.optimizer = "hyperopt"
optimizer_2.num_simulations = 1

optimizer_3 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_3["optimization_params"])
optimizer_3.optimizer = "hyperopt"
optimizer_3.num_simulations = 1

optimizer_4 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_4["optimization_params"])
optimizer_4.optimizer = "hyperopt"
optimizer_4.num_simulations = 1

optimizer_5 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_5["optimization_params"])
optimizer_5.optimizer = "hyperopt"
optimizer_5.num_simulations = 1

optimizer_6 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_6["optimization_params"])
optimizer_6.optimizer = "hyperopt"
optimizer_6.num_simulations = 1

optimizer_7 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_7["optimization_params"])
optimizer_7.optimizer = "hyperopt"
optimizer_7.num_simulations = 1

optimizer_8 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_8["optimization_params"])
optimizer_8.optimizer = "hyperopt"
optimizer_8.num_simulations = 1

optimizer_9 = mf.construct_optimizer(model_index, pid, optimization_criterion, other_params_9["optimization_params"])
optimizer_9.optimizer = "hyperopt"
optimizer_9.num_simulations = 1

prior_space = optimizer_2.get_prior()
prior_sample = hyperopt.pyll.stochastic.sample(prior_space)


# 1 - Model 1.1
# 2 - Model 2.1
# 4 - Model 1.2
# 5 - Model 2.2

# 1 - Model 3.0
# 2 - Model 3.1.1
# 4 - Model 3.1.2
# 5 - Model 3.2
# 7 - Model 3.3

agents_to_run = [
    "1",
    #"1_n",
    "2",
    #"3",
    "4",
    "5",
    "7",
    #"8",
    #"9"
]

plot_figures = [1, 2, 3,  7, 8]
def plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker):
    if 1 in plot_figures:
        plt.figure(1)
        plt.plot(range(len(plot_agent.bounds)), plot_agent.bounds, **plot_args_line)
    if 2 in plot_figures:
        plt.figure(2)
        plt.plot(range(len(plot_agent.precisions)), plot_agent.precisions, **plot_args_line)
    if 3 in plot_figures:
        plt.figure(3)
        plt.plot(range(len(plot_agent.covariance_over_time)), plot_agent.covariance_over_time, **plot_args_line)
    likelihood_times = plot_agent.action_likelihood_times
    last_rewards = [time["last_reward"] for time in likelihood_times]
    time_s = [time["time"] for time in likelihood_times]
    probs = [time["prob"] for time in likelihood_times]
    if 5 in plot_figures:
        plt.figure(5)
        plt.scatter(last_rewards, time_s, **plot_args_marker)
    if 6 in plot_figures:
        plt.figure(6)
        plt.scatter(probs, time_s, **plot_args_marker)
    if 7 in plot_figures:
        plt.figure(7)
        plt.plot(range(len(time_s)), time_s, **plot_args_line)
    if 8 in plot_figures:
        plt.figure(8)
        plt.scatter(range(len(probs)), probs, **plot_args_marker)

if "1" in agents_to_run:
    start = time.time()
    print("Starting agent 1")
    rel, sim, agent_1 = optimizer_1.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 1: {0:0.3f}s".format(end - start))
    plot_agent = agent_1
    plot_args_line = {
        'color': 'blue',
        'label': '3.0, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_args_marker = {
        'color': 'blue',
        'label': '3.0, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "1_n" in agents_to_run:
    start = time.time()
    rel, sim, agent_1_n = optimizer_1_n.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    plot_agent = agent_1_n
    plot_args_line = {
        'color': 'purple',
        'label': 'Null, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_args_marker = {
        'color': 'purple',
        'label': 'Null, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated),
        'marker': '^'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "2" in agents_to_run:
    start = time.time()
    print("Starting agent 2")
    rel, sim, agent_2 = optimizer_2.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 2: {0:0.3f}s".format(end - start))
    plot_agent = agent_2
    plot_args_line = {
        'color': 'orange',
        'label': '3.1.1, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_args_marker = {
        'color': 'orange',
        'label': '3.1.1, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated),
        'marker': 'x'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "3" in agents_to_run:
    start = time.time()
    print("Starting agent 3")
    rel, sim, agent_3 = optimizer_3.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 3: {0:0.3f}s".format(end - start))
    plot_args = {
        'color': 'orange',
        'label': 'No LFA, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_mcrl_agent(plot_agent, plot_args)
    plt.figure(1)
    plt.plot(range(len(agent_3.bounds)), agent_3.bounds, color='red', linestyle='--', label='LFA0.05, {0:0.3f}, {1}'.format(end - start, agent_3.num_actions_updated))
    plt.figure(2)
    plt.plot(range(len(agent_3.precisions)), agent_3.precisions, color='red', linestyle='--', label='LFA0.05, {0:0.3f}, {1}'.format(end - start, agent_3.num_actions_updated))
    plt.figure(3)
    plt.plot(range(len(agent_3.covariance_over_time)), agent_3.covariance_over_time, color='red', linestyle='--', label='LFA0.05, {0:0.3f}, {1}'.format(end - start, agent_3.num_actions_updated))

if "4" in agents_to_run:
    start = time.time()
    print("Starting agent 4")
    rel, sim, agent_4 = optimizer_4.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 4: {0:0.3f}s".format(end - start))
    plot_agent = agent_4
    plot_args_line = {
        'color': 'green',
        'linestyle': '--',
        'label': '3.1.2, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_args_marker = {
        'color': 'green',
        'label': '3.1.2, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated),
        'marker': 'x'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "5" in agents_to_run:
    start = time.time()
    print("Starting agent 5")
    rel, sim, agent_5 = optimizer_5.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 5: {0:0.3f}s".format(end - start))
    plot_agent = agent_5
    plot_args_line = {
        'color': 'black',
        'label': '3.2, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated)
    }
    plot_args_marker = {
        'color': 'black',
        'label': '3.2, {0:0.3f}, {1}'.format(end - start, plot_agent.num_actions_updated),
        'marker': 'x'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "7" in agents_to_run:
    start = time.time()
    print("Starting agent 7")
    rel, sim, agent_7 = optimizer_7.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    print("Finished agent 7: {0:0.3f}s".format(end - start))
    plot_agent = agent_7
    plot_args_line = {
        'color': 'purple',
        'label': '3.3, {0:0.3f}s, {1}'.format(end - start, other_params_7["optimization_params"]["max_integration_degree"]),
        'linestyle': ':'
    }
    plot_args_marker = {
        'color': 'purple',
        'label': '3.3, {0:0.3f}s, {1}'.format(end - start, other_params_7["optimization_params"]["max_integration_degree"]),
        'marker': 'v'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "8" in agents_to_run:
    start = time.time()
    rel, sim, agent_8 = optimizer_8.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    plot_agent = agent_8
    plot_args_line = {
        'color': 'blue',
        'label': 'No LFA, {0:0.3f}s, {1}'.format(end - start, other_params_8["optimization_params"]["max_integration_degree"]),
        'linestyle': '--'
    }
    plot_args_marker = {
        'color': 'blue',
        'label': 'No LFA, {0:0.3f}s, {1}'.format(end - start, other_params_8["optimization_params"]["max_integration_degree"]),
        'marker': 'x'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)

if "9" in agents_to_run:
    start = time.time()
    rel, sim, agent_9 = optimizer_9.fit_with_params(optimization_criterion, prior_sample, True)
    end = time.time()
    plot_agent = agent_9
    plot_args_line = {
        'color': 'green',
        'label': 'No LFA, {0:0.3f}s, {1}'.format(end - start, other_params_9["optimization_params"]["max_integration_degree"]),
        'linestyle': '-.'
    }
    plot_args_marker = {
        'color': 'green',
        'label': 'No LFA, {0:0.3f}s, {1}'.format(end - start, other_params_9["optimization_params"]["max_integration_degree"]),
        'marker': '^'
    }
    plot_mcrl_agent(plot_agent, plot_args_line, plot_args_marker)



if "2" in agents_to_run and "1_n" in agents_to_run:
    plt.figure(4)
    prec_diff = np.array(agent_2.precisions) - np.array(agent_1_n.precisions)
    plt.plot(range(len(agent_1_n.precisions)), prec_diff, label='Precision')
    bounds_diff = np.array(agent_2.bounds) - np.array(agent_1_n.bounds)
    plt.plot(range(len(agent_1_n.bounds)), bounds_diff, label='bounds')
    plt.legend()
    plt.title("Difference in Bounds and Precisions Between 2 and 1_n")
    plt.xlabel("Action #")

# if ("1" in agents_to_run or "2" in agents_to_run) and not set(agents_to_run).isdisjoint({"7", "8", "9"}):
#     plt.figure(10)
#     plt.title('Difference between computed action likelihoods')
#     plt.xlabel("Action #")
#     plt.ylabel("Difference in computed probability")
#     times_2 = agent_2.action_likelihood_times
#     probs_2 = np.array([time["prob"] for time in times_2])
#     plt.plot(range(len(probs_2)), probs_2 - probs_2, color='orange', label='No LFA, D=10, Diff={0:0.3f}'.format(
#         np.sum(np.abs(probs_2-probs_2))
#     ))
#
#     if "7" in agents_to_run:
#         times_7 = agent_7.action_likelihood_times
#         probs_7 = np.array([time["prob"] for time in times_7])
#         plt.plot(range(len(probs_2)), probs_7 - probs_2, color='purple', label='No LFA, D={0}, Diff={1:0.3f}'.format(
#             other_params_7["optimization_params"]["max_integration_degree"],
#             np.sum(np.abs(probs_7-probs_2))
#         ))
#
#     if "8" in agents_to_run:
#         times_8 = agent_8.action_likelihood_times
#         probs_8 = np.array([time["prob"] for time in times_8])
#         plt.plot(range(len(probs_2)), probs_8 - probs_2, color='blue', label='No LFA, D={0}, Diff={1:0.3f}'.format(
#             other_params_8["optimization_params"]["max_integration_degree"],
#             np.sum(np.abs(probs_8-probs_2))
#         ))
#
#     if "9" in agents_to_run:
#         times_9 = agent_9.action_likelihood_times
#         probs_9 = np.array([time["prob"] for time in times_9])
#         plt.plot(range(len(probs_2)), probs_9 - probs_2, color='green', label='LFA, D={0}, Diff={1:0.3f}'.format(
#             other_params_9["optimization_params"]["max_integration_degree"],
#             np.sum(np.abs(probs_9-probs_2))
#         ))
#
#     plt.legend()

if 1 in plot_figures:
    plt.figure(1)
    plt.xlabel("Action #")
    plt.ylabel("Bounds of likelihood integral")
    plt.legend()

if 2 in plot_figures:
    plt.figure(2)
    plt.xlabel("Action #")
    plt.ylabel("Sum of param precision")
    plt.legend()

if 3 in plot_figures:
    plt.figure(3)
    plt.xlabel("Action #")
    plt.ylabel("Sum of covariance diagonal")
    plt.legend()

if 5 in plot_figures:
    plt.figure(5)
    plt.xlabel("Last Reward")
    plt.ylabel("Time Taken for L computation")
    plt.legend()

if 6 in plot_figures:
    plt.figure(6)
    plt.xlabel("Action Probability")
    plt.ylabel("Time Taken for L computation")
    plt.legend()

if 7 in plot_figures:
    plt.figure(7)
    plt.xlabel("Action #")
    plt.ylabel("Time taken for L computation")
    plt.legend()

if 8 in plot_figures:
    plt.figure(8)
    plt.xlabel("Action #")
    plt.ylabel("Action Probability")
    plt.legend()

"""

plt.figure()



# [{
#     #: { time: #, prob: #}
#     selected: #
# }]


prob_time_list_1 = []
for dict_ in times_1:
    prob_time_list_1 += [value for (key, value) in dict_.items() if str.isnumeric(str(key))]

time_list_1 = [dict_["time"] for dict_ in prob_time_list_1]
prob_list_1 = [dict_["prob"] for dict_ in prob_time_list_1]

prob_time_list_2 = []
for dict_ in times_2:
    prob_time_list_2 += [value for (key, value) in dict_.items() if str.isnumeric(str(key))]

time_list_2 = [dict_["time"] for dict_ in prob_time_list_2]
prob_list_2 = [dict_["prob"] for dict_ in prob_time_list_2]

plt.scatter(prob_list_1, time_list_1, color='blue', label="LFA")
plt.scatter(prob_list_2, time_list_2, color='orange', label="No LFA")
plt.legend()
"""

plt.show()