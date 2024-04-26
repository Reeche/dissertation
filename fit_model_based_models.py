from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
import pickle
import sys
from pathlib import Path

"""
There are 4 variants: 
full: every node has its own alpha and beta
linear: alpha and beta are linearly interpolated between the node levels
uniform: alpha and beta are the same for all nodes
level: alpha and beta are the same for all nodes of the same level

Currently, the full model works the best and uniform the worst
Could not tell differences between the linear and level models 
"""

def plot_score(res, participant, pid, exp_name):
    plt.plot(np.mean(res["rewards"], axis=0), color="r", label="Model")
    # plt.plot(participant["r"], color="b", label="Participant")
    plt.legend()
    plt.title(f"Score")
    plt.show()
    # plt.savefig(f"results_mb_2000_inc/mcrl/{exp_name}_mb/plots/score_{pid}.png")
    plt.close()
    return None

def plot_mer(res, participant, pid, exp_name):
    plt.plot(np.mean(res["mer"], axis=0), color="r", label="Model")
    plt.legend()
    plt.title(f"MER")
    plt.show()
    # plt.savefig(f"results_mb_2000_inc/mcrl/{exp_name}_mb/plots/score_{pid}.png")
    plt.close()
    return None


def plot_clicks(res, participant):
    pid_action = []
    for action in participant["a"]:
        pid_action.append(len(action) - 1)
    model_action = []
    for action in res["a"][0]:
        model_action.append(len(action) - 1)
    plt.plot(pid_action, color="r", label="Participant")
    plt.plot(model_action, color="b", label="Model")
    plt.legend()
    plt.title(f"Clicks")
    plt.show()
    # plt.savefig(f"results_mb_2000_inc/mcrl/{exp_name}_mb/plots/clicks_{pid}.png")
    plt.close()
    return None


def cost_function(depth):
    if depth == 0:
        return 0
    if depth == 1:
        return 1
    if depth == 2:
        return 3
    if depth == 3:
        return 30


if __name__ == "__main__":
    exp_name = sys.argv[1]
    criterion = sys.argv[2]
    pid = int(sys.argv[3])
    model_variant = sys.argv[4]

    """
    List of models: 
    
    no assumption: Beta(1,1) for all nodes
    uniform assumption: Beta(alpha, beta) for all nodes
    level assumption: Beta(alpha_level, beta_level) for all nodes of the same level    
    
    ways of update: 
    individual node: update every node
    level update: update all nodes of the same level
    """

    # exp_name = "high_variance_low_cost"  # "strategy_discovery
    # criterion = "likelihood"  # "number_of_clicks_likelihood"
    # pid = 4
    # model_variant = "full" #"full", "linear", "uniform", "level", "no_assumption"

    E = Experiment(exp_name, data_path=f"results_mb_test12/mcrl/{exp_name}_mb")

    if exp_name == "high_variance_high_cost" or exp_name == "low_variance_high_cost":
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": click_cost
    }

    if exp_name == "strategy_discovery":
        number_of_trials = 120
    else:
        number_of_trials = 35

    if criterion != "likelihood":
        num_simulations = 30
    else:
        num_simulations = 1

    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=f"results_mb_test12/mcrl/{exp_name}_mb",
        number_of_trials=number_of_trials)

    pid_context, env = mf.get_participant_context(pid)
    participant_obj = ParticipantIterator(pid_context, click_cost=click_cost)

    # todo: need to choose a sensible range that takes the click cost into consideration
    # Has to be symmetric, otherwise biased towards negative expected value
    if exp_name in ["v1.0", "c2.1"]:
        # value_range = list(range(-49, 49))
        value_range = [-48, -24, -8, -4, -2, 2, 4, 8, 24, 48]
        term_range = list(range(-61, 61))
    elif exp_name == "c1.1":
        value_range = list(range(-31, 31))
    elif exp_name in ["high_variance_high_cost", "high_variance_low_cost"]:
        value_range = [-1000, -100, -50, -20, 50, 100]
        term_range = list(range(-3001, 3001))
    elif exp_name in ["low_variance_high_cost", "low_variance_low_cost"]:
        value_range = list(range(-7, 7))
    elif exp_name == "strategy_discovery":
        value_range = list(range(-51, 51))
    else:
        raise ValueError("Experiment name not recognised")

    model = ModelBased(env, value_range, term_range, participant_obj, criterion, num_simulations, model_variant, compute_likelihood=True)

    if criterion != "likelihood": #todo
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', -100, 100),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3)),
            'dist_alpha': hp.uniform('dist_alpha', 0, 5),
            'dist_beta': hp.uniform('dist_beta', 0, 5),
        }
    else:
        min_value = 1
        max_value = 10
        if model_variant == "uniform":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', 0.01, 10),
                'alpha': hp.uniform('alpha', min_value, max_value),
                'beta': hp.uniform('beta', min_value, max_value),
                'click_weight': hp.uniform('click_weight', 1, 100),
            }
        elif model_variant == "linear": #not really used
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', -100, 100),
                'alpha_weight': hp.uniform('alpha_weight', np.log(min_value), np.log(max_value)),
                'beta_weight': hp.uniform('beta_weight', np.log(min_value), np.log(max_value)),
                'alpha_intercept': hp.uniform('alpha_intercept', np.log(min_value), np.log(max_value)),
                'beta_intercept': hp.uniform('beta_intercept', np.log(min_value), np.log(max_value)),
            }
        elif model_variant == "full" or model_variant == "level":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', 0.01, 10),
                'alpha_1': hp.uniform('alpha_1', min_value, max_value),
                'beta_1': hp.uniform('beta_1', min_value, max_value),
                'alpha_2': hp.uniform('alpha_2', min_value, max_value),
                'beta_2': hp.uniform('beta_2', min_value, max_value),
                'alpha_3': hp.uniform('alpha_3', min_value, max_value),
                'beta_3': hp.uniform('beta_3', min_value, max_value),
                'click_weight': hp.uniform('click_weight', 1, 100),
            }
        elif model_variant == "no_assumption":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', 0.01, 10),
                'click_weight': hp.uniform('click_weight', 1, 100),
            }
        else:
            raise ValueError(f"Model not recognised: {model_variant}")

    trials = True
    trials = Trials() if trials else None
    best_params = fmin(fn=model.run_multiple_simulations,
                       space=fspace,
                       algo=tpe.suggest,
                       max_evals=2000,
                       show_progressbar=True)

    ## simulate using the best parameters
    model.compute_likelihood = False

    # best_params = {
    #     'inverse_temp': 1,
    #     'alpha_1': 1,
    #     'beta_1': 1,
    #     'alpha_2': 1,
    #     'beta_2': 1,
    #     'alpha_3': 1,
    #     'beta_3': 1,
    #     'click_weight': 10,
    # }

    model.env.reset()
    model.participant_obj.reset()
    res = model.run_multiple_simulations(best_params)

    ## save result and best parameters
    res.update(best_params)

    ## check if dir exist
    if not Path(f"results_mb_test12/mcrl/{exp_name}_mb").exists():
        Path(f"results_mb_test12/mcrl/{exp_name}_mb").mkdir(parents=True, exist_ok=True)

    output = open(f'results_mb_test12/mcrl/{exp_name}_mb/{pid}_{criterion}_{model_variant}.pkl', 'wb')
    pickle.dump(res, output)
    output.close()

    # plot_score(res, model.p_data, pid, exp_name)
    # plot_mer(res, model.p_data, pid, exp_name)
    # plot_clicks(res, model.p_data)
    # print(res)



