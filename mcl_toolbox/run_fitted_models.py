from mcl_toolbox.global_vars import *
from mcl_toolbox.utils.learning_utils import pickle_load, create_dir
from mcl_toolbox.utils.model_utils import ModelFitter

if __name__ == "__main__":
    exp_name = "T1.1"
    exp_attributes = {'block': 'test'}
    model_index = 1825
    pid = 4
    num_simulations = 4
    plotting = True

    sim_params = {'num_simulations': num_simulations}
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

    mf = ModelFitter(exp_name, exp_attributes=exp_attributes)
    (res, prior) = pickle_load(param_dir.joinpath(f"{pid}_{fit_criterion}_{model_index}.pkl"))

    mf.simulate_params(
        model_index,
        res[0],
        pid=pid,
        sim_params=sim_params,
        sim_dir=model_info_directory,
        plot_dir=plot_directory,
    )