from mcl_toolbox.utils.participant_utils import ParticipantIterator
from models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from hyperopt import hp, fmin, tpe, Trials
import matplotlib.pyplot as plt


def plot_score(model_results, participant):
    plt.plot(model_results['r'], label="Model")
    plt.plot(participant.score, label="Participant")
    plt.legend()
    plt.show()
    return None


if __name__ == "__main__":
    exp_name = "v1.0"
    pid = 1

    E = Experiment("v1.0")
    p = E.participants[pid]
    participant_obj = ParticipantIterator(p)

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": 1,
        "learn_from_path": True,
    }


    def cost_function(depth):
        if depth == 0:
            return 0
        if depth == 1:
            return -1
        if depth == 2:
            return -3
        if depth == 3:
            return -30


    if exp_name == "high_variance_high_cost" or exp_name == "low_variance_high_cost":
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    number_of_trials = 35

    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=None,
        number_of_trials=number_of_trials)

    pid, env = mf.get_participant_context(pid)

    # todo: need to choose a sensible range that takes the click cost into consideration
    value_range = list(range(-100, 100))

    model = ModelBased(pid, env, value_range, True, participant_obj)
    # res = model.simulate(compute_likelihood=True, participant=participant_obj)
    # print(res)
    fspace = {
        'inverse_temp': hp.uniform('inverse_temp', 0, 1)
    }

    # minimize the objective over the space
    best_params = fmin(fn=model.simulate,
                space=fspace,
                algo=tpe.suggest,
                max_evals=5,
                # trials=True,
                show_progressbar=True)

    ## simulate using the best parameters
    res = model.simulate(best_params)
    print(best_params)
    plot_score(res, pid)
