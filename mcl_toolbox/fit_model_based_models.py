from mcl_toolbox.utils.participant_utils import ParticipantIterator
from models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from hyperopt import hp, fmin, tpe, Trials
import matplotlib.pyplot as plt
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab


def plot_score(res, participant):
    plt.plot(res['rewards'], color="r", label="Model")
    plt.plot(participant.score, color="b", label="Participant")
    plt.legend()
    plt.show()
    # plt.savefig(f"../results/mcrl/{exp_name}_model_based/plots/{participant.pid}.png")
    plt.close()
    return None


if __name__ == "__main__":
    exp_name = "v1.0"
    pid = 1  # 35 is okay adaptive  # 51#1 pid 1 is very adaptive

    E = Experiment("v1.0")
    p = E.participants[pid]
    participant_obj = ParticipantIterator(p)


    def cost_function(depth):
        if depth == 0:
            return 0
        if depth == 1:
            return -1
        if depth == 2:
            return -3
        if depth == 3:
            return -30


    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": 1,
        "learn_from_path": True,
    }

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
    value_range = list(range(-120, 120))

    model = ModelBased(pid, env, value_range, True, participant_obj)
    # res = model.simulate(compute_likelihood=True, participant=participant_obj)

    fspace = {
        'inverse_temp': hp.uniform('inverse_temp', 0, 1)
    }

    # minimize the objective over the space
    best_params = fmin(fn=model.simulate,
                       space=fspace,
                       algo=tpe.suggest,
                       max_evals=10,
                       # trials=True,
                       show_progressbar=True)

    # best_params = {'inverse_temp': 0.5}
    ## simulate using the best parameters
    model.compute_likelihood = False
    res = model.simulate(best_params)
    # print(res)

    ## save result and best parameters
    # res.update(best_params)
    # output = open(f'../results/mcrl/{exp_name}_model_based/{pid}.pkl', 'wb')
    # pickle.dump(res, output)
    # output.close()
    plot_score(res, pid)
