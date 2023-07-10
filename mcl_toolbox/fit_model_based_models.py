from mcl_toolbox.utils.participant_utils import ParticipantIterator
from models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from hyperopt import hp, fmin, tpe
import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_score(res, participant):
    plt.plot(np.mean(res["mer"], axis=0), color="r", label="Model")
    plt.plot(participant["mer"], color="b", label="Participant")
    plt.legend()
    plt.show()
    # plt.savefig(f"../results/mcrl/{exp_name}_model_based/plots/{participant.pid}.png")
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
    exp_name = "v1.0"
    E = Experiment("v1.0")
    # pid_dict = {
    #     'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75,
    #              77,
    #              80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
    #              148, 150, 154, 155, 158, 160, 165, 169, 173],
    #     'c2.1': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
    #              79, 84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138,
    #              142, 145, 149, 152, 156, 162, 164, 166, 170, 172],
    #     'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
    #              83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
    #              151, 153, 157, 159, 161, 163, 167, 168, 171]}
    # pid = 1  # 35 is okay adaptive  # 51#1 pid 1 is very adaptive

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": 1
    }

    number_of_trials = 35
    criterion = "number_of_clicks_likelihood"
    if criterion != "likelihood":
        num_simulations = 2
    else:
        num_simulations = 1

    if exp_name == "high_variance_high_cost" or exp_name == "low_variance_high_cost":
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    # for pid in pid_dict[exp_name]:
    for pid in [1]:
        p = E.participants[pid]
        participant_obj = ParticipantIterator(p)

        mf = ModelFitter(
            exp_name=exp_name,
            exp_attributes=exp_attributes,
            data_path=None,
            number_of_trials=number_of_trials)

        pid_context, env = mf.get_participant_context(pid)

        # todo: need to choose a sensible range that takes the click cost into consideration
        value_range = list(range(-120, 120))

        model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, test_fitted_model=False)
        # res = model.simulate(compute_likelihood=True, participant=participant_obj)

        fspace = {
            'inverse_temp': hp.uniform('inverse_temp', 0, 1),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3))
        }

        # minimize the objective over the space
        best_params = fmin(fn=model.run_multiple_simulations,
                           space=fspace,
                           algo=tpe.suggest,
                           max_evals=1,
                           # trials=True,
                           show_progressbar=True)

        ## simulate using the best parameters
        model.test_fitted_model = True
        res = model.run_multiple_simulations(best_params)
        print(res)

        ## save result and best parameters
        # res.update(best_params)
        # output = open(f'../results/mcrl/{exp_name}_model_based/data/{pid}.pkl', 'wb')
        # pickle.dump(res, output)
        # output.close()
        plot_score(res, model.p_data)

