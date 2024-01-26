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
    exp_name = "strategy_discovery"

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
    model_index = 1743
    pid = 38
    num_simulations = 1
    plotting = True

    sim_params = {"num_simulations": num_simulations}
    fit_criterion = "likelihood"

    parent_directory = Path(__file__).parents[1]
    param_dir = parent_directory.joinpath(f"results/mcrl/{exp_name}_priors")

    # and directory to save fit model info in
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
        number_of_trials=1200,
    )

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

    plot_score(sim_data)
    print(r_data)
    print(sim_data)
