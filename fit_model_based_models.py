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
    # model_variant = sys.argv[4]

    # exp_name = "strategy_discovery"  # "strategy_discovery
    # criterion = "likelihood"  # "number_of_clicks_likelihood"
    # pid = 38
    model_variant = "full" #"full", "linear", "uniform", "level"

    E = Experiment(exp_name, data_path=f"results_mb_2000_v2/mcrl/{exp_name}_mb")

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
        number_of_trials = 1200
    else:
        number_of_trials = 35

    if criterion != "likelihood":
        num_simulations = 30
    else:
        num_simulations = 1

    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=f"results_mb_2000_v2/mcrl/{exp_name}_mb",
        number_of_trials=number_of_trials)

    pid_context, env = mf.get_participant_context(pid)
    participant_obj = ParticipantIterator(pid_context, click_cost=click_cost)

    # todo: need to choose a sensible range that takes the click cost into consideration
    # Has to be symmetric, otherwise biased towards negative expected value
    if exp_name in ["v1.0", "c2.1"]:
        value_range = list(range(-48, 49))
    elif exp_name == "c1.1":
        value_range = list(range(-30, 31))
    elif exp_name in ["high_variance_high_cost", "high_variance_low_cost"]:
        value_range = list(range(-1000, 1001))
    elif exp_name in ["low_variance_high_cost", "low_variance_low_cost"]:
        value_range = list(range(-6, 7))
    elif exp_name == "strategy_discovery":
        value_range = list(range(-50, 51))
    else:
        raise ValueError("Experiment name not recognised")

    model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, model_variant, compute_likelihood=True)

    if criterion != "likelihood": #todo
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', -100, 100),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3)),
            'dist_alpha': hp.uniform('dist_alpha', 0, 5),
            'dist_beta': hp.uniform('dist_beta', 0, 5),
        }
    else:
        if model_variant == "uniform":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', -100, 100),
                'dist_alpha': hp.uniform('dist_alpha', np.log(1e-8), np.log(5)),
                'dist_beta': hp.uniform('dist_beta', np.log(1e-8), np.log(5)),
            }
        elif model_variant == "linear":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', -100, 100),
                'alpha_weight': hp.uniform('alpha_weight', np.log(1e-8), np.log(10)),
                'beta_weight': hp.uniform('beta_weight', np.log(1e-8), np.log(10)),
                'alpha_intercept': hp.uniform('alpha_intercept', np.log(1e-8), np.log(5)),
                'beta_intercept': hp.uniform('beta_intercept', np.log(1e-8), np.log(5)),
            }
        elif model_variant == "full":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', -1, 1),
                'alpha_1': hp.uniform('alpha_1', np.log(1), np.log(5)),
                'beta_1': hp.uniform('beta_1', np.log(1), np.log(5)),
                'alpha_2': hp.uniform('alpha_2', np.log(1), np.log(5)),
                'beta_2': hp.uniform('beta_2', np.log(1), np.log(5)),
                'alpha_3': hp.uniform('alpha_3', np.log(1), np.log(5)),
                'beta_3': hp.uniform('beta_3', np.log(1), np.log(5)),
                'click_weight': hp.uniform('click_weight', 1, 50),
            }
        elif model_variant == "level":
            fspace = {
                'inverse_temp': hp.uniform('inverse_temp', -100, 100),
                'alpha_1': hp.uniform('alpha_1', np.log(1), np.log(5)),
                'beta_1': hp.uniform('beta_1', np.log(1), np.log(5)),
                'alpha_2': hp.uniform('alpha_2', np.log(1), np.log(5)),
                'beta_2': hp.uniform('beta_2', np.log(1), np.log(5)),
                'alpha_3': hp.uniform('alpha_3', np.log(1), np.log(5)),
                'beta_3': hp.uniform('beta_3', np.log(1), np.log(5)),
                'click_weight': hp.uniform('click_weight', 1, 50),
            }
        else:
            raise ValueError(f"Model not recognised: {model_variant}")

    trials = True
    trials = Trials() if trials else None
    best_params = fmin(fn=model.run_multiple_simulations,
                       space=fspace,
                       algo=tpe.suggest,
                       max_evals=2000,
                       show_progressbar=False)

    ## simulate using the best parameters
    model.compute_likelihood = False

    # for pid 38 strategy discovery
    # best_params = {'inverse_temp': 0.33,
    #                'alpha_1': np.log(0.4404),
    #                'beta_1': np.log(0.7824),
    #                'alpha_2': np.log(0.1946),
    #                'beta_2': np.log(1.5247),
    #                'alpha_3': np.log(1.6084),
    #                'beta_3': np.log(0.0022),
    #                'click_weight': 49}

    model.env.reset()
    model.participant_obj.reset()
    res = model.run_multiple_simulations(best_params)

    ## save result and best parameters
    res.update(best_params)

    ## check if dir exist
    if not Path(f"results_mb_2000_v2/mcrl/{exp_name}_mb").exists():
        Path(f"results_mb_2000_v2/mcrl/{exp_name}_mb").mkdir(parents=True, exist_ok=True)

    output = open(f'results_mb_2000_v2/mcrl/{exp_name}_mb/{pid}_{criterion}_{model_variant}.pkl', 'wb')
    pickle.dump(res, output)
    output.close()

    # plot_score(res, model.p_data, pid, exp_name)
    # plot_clicks(res, model.p_data)
    # print(res)



