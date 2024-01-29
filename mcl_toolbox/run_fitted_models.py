from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from mcl_toolbox.utils.learning_utils import create_dir, pickle_load
from mcl_toolbox.utils.model_utils import ModelFitter


def plot_score(res):
    plt.plot(np.mean(res["r"], axis=0), color="r", label="Model")
    plt.legend()
    plt.show()
    # plt.savefig(f"results_mb_2000_inc/mcrl/{exp_name}_mb/plots/score_{pid}.png")
    plt.close()
    return None


def plot_num_clicks(data, model):
    # get the number of clicks by looking at the length of each sublist in res["a"]
    # init dict with key = trial, value = 0
    model_num_clicks = []
    for pid, value in data.items():
        num_clicks = {trial: 0 for trial in range(35)}
        for simulation in value["a"]:
            for trial, sublist in enumerate(simulation):
                num_clicks[trial] += len(sublist) - 1

        # for every value in num_clicks, divide by num_simulations
        num_simulations = len(value["a"])
        for trial in num_clicks:
            num_clicks[trial] /= num_simulations
        model_num_clicks.append(list(num_clicks.values()))

    plt.plot(np.mean(model_num_clicks, axis=0), color="r", label=model)
    plt.legend()
    plt.show()
    # plt.savefig(f"{model}_num_clicks.png")
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
    exp_name = "high_variance_low_cost"

    if exp_name == "high_variance_high_cost" or exp_name == "low_variance_high_cost":
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    exp_attributes = {
        "exclude_trials": None,  # Trials to be excluded
        "block": None,  # Block of the experiment
        "experiment": None,  # Experiment object can be passed directly with
        "click_cost": click_cost
    }
    model_index = 1756
    pid = 7 #7
    num_simulations = 10
    plotting = False

    sim_params = {"num_simulations": num_simulations,
                  "click_cost": click_cost}
    fit_criterion = "likelihood"

    parent_directory = Path(__file__).parents[1]
    # param_dir = parent_directory.joinpath(f"results_mf_models_2000/mcrl_backup_correct_ones/{exp_name}_priors")
    param_dir = parent_directory.joinpath(f"results/mcrl/{exp_name}_priors")
    # and directory to save fit model info in
    # model_info_directory = parent_directory.joinpath(f"results_mf_models_2000/mcrl/{exp_name}_data")
    model_info_directory = parent_directory.joinpath(f"results/mcrl/{exp_name}_data")
    create_dir(model_info_directory)

    # add directory for reward plots, if plotting
    plot_directory = None
    if plotting:
        plot_directory = parent_directory.joinpath(f"results/mcrl/{exp_name}_plots")
        create_dir(plot_directory)

    mf = ModelFitter(
        exp_name,
        exp_attributes=exp_attributes,
        data_path=f"results/mcrl/{exp_name}",
        number_of_trials=120,
    )

    all_data = {}
    # for pid in [4, 7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 141, 145, 146, 151,
    #             154, 158, 180, 189, 197]:
    (res, prior) = pickle_load(
        param_dir.joinpath(f"{pid}_{fit_criterion}_{model_index}.pkl")
    )

    r_data, sim_data = mf.simulate_params(
        model_index,
        res[0],
        pid=pid,
        sim_params=sim_params,
        sim_dir=model_info_directory,
        plot_dir=plot_directory,
    )

    all_data[pid] = sim_data

    plot_num_clicks(all_data, model_index)
    # plot_score(sim_data)
    # print(r_data)
    # print(sim_data)
