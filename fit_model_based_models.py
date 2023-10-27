from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
import torch
import pickle
import sys


def plot_score(res, participant, pid, exp_name):
    plt.plot(np.mean(res["rewards"], axis=0), color="r", label="Model")
    plt.plot(participant["mer"], color="b", label="Participant")
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
        return -1
    if depth == 2:
        return -3
    if depth == 3:
        return -30


if __name__ == "__main__":
    # exp_name = sys.argv[1]
    # criterion = sys.argv[2]
    # pid = int(sys.argv[3])

    exp_name = "v1.0"  # "strategy_discovery
    criterion = "likelihood"  # "number_of_clicks_likelihood"
    pid = 17

    E = Experiment(exp_name, data_path=f"results_mb_2000_inc_bias/mcrl/{exp_name}_mb")

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

    p = E.participants[pid]
    participant_obj = ParticipantIterator(p)

    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=f"results_mb_2000_inc_bias/mcrl/{exp_name}_mb",
        number_of_trials=number_of_trials)

    pid_context, env = mf.get_participant_context(pid)

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

    model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, test_fitted_model=False)

    if criterion != "likelihood":
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', -1000, 1000),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3)),
            'dist_alpha': hp.uniform('dist_alpha', 0, 10),
            'dist_beta': hp.uniform('dist_beta', 0, 10),
            'bias': hp.uniform('bias', -2, 2)
        }
    else:
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', -100, 100),
            'dist_alpha': hp.uniform('dist_alpha', 0, 10),
            'dist_beta': hp.uniform('dist_beta', 0, 10),
            'bias': hp.uniform('bias', -5, 5)
        }

    trials = True
    trials = Trials() if trials else None
    best_params = fmin(fn=model.run_multiple_simulations,
                       space=fspace,
                       algo=tpe.suggest,
                       max_evals=2,
                       show_progressbar=True)

    ## simulate using the best parameters
    model.test_fitted_model = True

    # best_params = {'inverse_temp': torch.tensor(1),
    #                'dist_alpha': torch.tensor(1),
    #                'dist_beta': torch.tensor(1),
    #                'bias': torch.tensor(1.3)}
    # model.init_model_params(best_params['dist_alpha'], best_params['dist_beta'])

    model.env.reset()
    model.participant_obj.reset()
    res = model.run_multiple_simulations(best_params)

    ## save result and best parameters
    res.update(best_params)
    # plot_score(res, model.p_data, pid, exp_name)
    # plot_clicks(res, model.p_data)
    print(res)

    # output = open(f'results_mb_2000_inc_bias/mcrl/{exp_name}_mb/{pid}_{criterion}.pkl', 'wb')
    # pickle.dump(res, output)
    # output.close()
