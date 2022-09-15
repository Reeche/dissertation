# from mcl_toolbox.global_vars import *
from pathlib import Path

from mcl_toolbox.utils.learning_utils import create_dir, pickle_load
from mcl_toolbox.utils.model_utils import ModelFitter

if __name__ == "__main__":
    exp_name = "F1"
    exp_attributes = {
        "exclude_trials": None,  # Trials to be excluded
        "block": None,  # Block of the experiment
        "experiment": None  # Experiment object can be passed directly with
        # pipeline and normalized features attached
    }
    model_index = 1729
    pid = 0
    num_simulations = 31
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
        exp_name, exp_attributes=exp_attributes, number_of_trials=num_simulations
    )
    (res, prior) = pickle_load(
        param_dir.joinpath(f"{pid}_{fit_criterion}_{model_index}.pkl")
    )

    mf.simulate_params(
        model_index,
        res[0],
        pid=pid,
        sim_params=sim_params,
        sim_dir=model_info_directory,
        plot_dir=plot_directory,
    )
