from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from hyperopt import hp, fmin, tpe
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch


def plot_score(res, participant, pid, exp_name):
    plt.plot(np.mean(res["mer"], axis=0), color="r", label="Model")
    plt.plot(participant["mer"], color="b", label="Participant")
    plt.legend()
    plt.show()
    # plt.savefig(f"results/mcrl/{exp_name}_model_based/plots/{pid}.png")
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
    # plt.savefig(f"results/mcrl/{exp_name}_mb/plots/{pid}.png")
    plt.close()
    return None


def cost_function(depth):
    if depth == 0:
        return 0
    if depth == 1:
        return -1
    if depth == 2:
        return -1
    if depth == 3:
        return -1


if __name__ == "__main__":
    # exp_name = sys.argv[1]
    # criterion = sys.argv[2]
    # pid = int(sys.argv[3])

    exp_name = "high_variance_low_cost"  # "strategy_discovery
    criterion = "likelihood"  # "number_of_clicks_likelihood"
    pid = 4  # 2

    E = Experiment(exp_name)

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

    number_of_trials = 35
    criterion = "likelihood"
    if criterion != "likelihood":
        num_simulations = 2
    else:
        num_simulations = 1

    p = E.participants[pid]
    participant_obj = ParticipantIterator(p)

    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=None,
        number_of_trials=number_of_trials)

    pid_context, env = mf.get_participant_context_model_based(pid)

    # todo: need to choose a sensible range that takes the click cost into consideration
    # Has to be symmetric, otherwise biased towards negative expected value
    value_range = list(range(-1001, 1001))

    model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, test_fitted_model=False)
    # res = model.simulate(compute_likelihood=True, participant=participant_obj)

    # the higher inverse temp, the less exploration!
    # the smaller, the more exploration!
    if criterion != "likelihood":
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', 1e-3, 1e3),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3)),
            'alpha_multiplier': hp.uniform('alpha_multiplier', 10, 100),
            'dist_alpha': hp.uniform('dist_alpha', 0, 10),
            'dist_beta': hp.uniform('dist_beta', 0, 10)
        }
    else:
        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', -10, 10),
            'alpha_multiplier': hp.uniform('alpha_multiplier', 1, 2),
            'dist_alpha': hp.uniform('dist_alpha', 1, 10),
            'dist_beta': hp.uniform('dist_beta', 1, 10)
        }

    ###minimize the objective over the space
    # best_params = fmin(fn=model.run_multiple_simulations,
    #                    space=fspace,
    #                    algo=tpe.suggest,
    #                    max_evals=1,
    #                    # trials=True,
    #                    show_progressbar=True,
    #                    rstate=np.random.default_rng(0))
    #
    # #{'alpha_multiplier': 13.287755104871485, 'dist_alpha': 3.886061271614582, 'dist_beta': 9.589713869950941,
    # # 'inverse_temp': -348.74003358438887}
    #
    # print(best_params)
    ## simulate using the best parameters
    model.test_fitted_model = True
    # have to use floats for log_softmax
    best_params = {'inverse_temp': torch.tensor(1.0),
                   'alpha_multiplier': torch.tensor(1.0),
                   'dist_alpha': torch.tensor(2.0), #alpha 2, beta 2 => inverse normal shape
                   'dist_beta': torch.tensor(2.0)}
    model.env.reset()
    model.participant_obj.reset()
    model.init_model_params(best_params['dist_alpha'], best_params['dist_beta'], best_params['alpha_multiplier'])
    model.init_distributions()
    res = model.run_multiple_simulations(best_params)

    ## save result and best parameters
    res.update(best_params)
    plot_score(res, model.p_data, pid, exp_name)
    plot_clicks(res, model.p_data)

    # output = open(f'results/mcrl/{exp_name}_model_based/data/{pid}_{criterion}.pkl', 'wb')
    # pickle.dump(res, output)
    # output.close()
