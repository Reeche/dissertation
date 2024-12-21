#! /lustre/work/rhe/mcl_toolbox/mcl_toolbox/mcl_toolbox_env/bin/python
import ast
import sys
import time
from pathlib import Path

from mcl_toolbox.utils.learning_utils import create_dir
from mcl_toolbox.utils.model_utils import ModelFitter

"""
Run this using:
python3 fit_mcrl_models.py <exp_name> <model_index> <optimization_criterion> <pid> <number_of_trials> <string of other parameters>
<optimization_criterion> can be ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]
Example: python3 mcl_toolbox/fit_mcrl_models.py v1.0 1 pseudo_likelihood 1 35 "{\"plotting\":True, \"optimization_params\" :{\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"
"""


# Likelihood computation is currently available for all reinforce and lvoc models but not for RSSL


def fit_model(
    exp_name,
    pid,
    number_of_trials,
    model_index,
    optimization_criterion,
    optimization_params=None,
    exp_attributes=None,
    sim_params=None,
    simulate=True,
    plotting=False,
    data_path="results/cm",
    save_path=None,
):
    """

    :param exp_name: experiment name, which is the folder name of the experiment in ../data/human/
    :param pid: participant id, as int
    :param model_index: model index, as displayed in rl_models.csv
    :param optimization_criterion: as string, choose one of: ["pseudo_likelihood", "mer_performance_error", "performance_error", "likelihood", "number_of_clicks"]
    :param optimization_params: parameters for ParameterOptimizer.optimize, passed in as a dict
    :return:
    """

    # create directory to save priors in
    if save_path is None:
        save_path = Path(__file__).resolve().parents[0].joinpath(f"results/mcrl")
    else:
        save_path.mkdir(parents=True, exist_ok=True)

    prior_directory = save_path.joinpath(f"{exp_name}_priors")
    create_dir(prior_directory)

    model_info_directory = None
    plot_directory = None
    if simulate:
        # and directory to save fit model info in
        model_info_directory = save_path.joinpath(f"{exp_name}_data")
        create_dir(model_info_directory)
        if plotting:
            # add directory for reward plots, if plotting
            plot_directory = save_path.joinpath(f"{exp_name}_plots")
            create_dir(plot_directory)

    mf = ModelFitter(
        exp_name,
        exp_attributes=exp_attributes,
        data_path=data_path,
        number_of_trials=number_of_trials,
    )
    res, prior, obj_fn = mf.fit_model(
        model_index,
        pid,
        optimization_criterion,
        optimization_params,
        params_dir=prior_directory,
    )
    if simulate:
        r_data, sim_data = mf.simulate_params(
            model_index,
            res[0],
            pid=pid,
            sim_dir=model_info_directory,
            plot_dir=plot_directory,
            # sim_params=sim_params,
            sim_params=optimization_params,
        )
    return r_data, sim_data


if __name__ == "__main__":
    # exp_name = sys.argv[1]
    # model_index = int(sys.argv[2])
    # optimization_criterion = sys.argv[3]
    # pid = int(sys.argv[4])
    # number_of_trials = int(sys.argv[5])
    # other_params = {"plotting": False}
    # if len(sys.argv) > 6:
    #     other_params = ast.literal_eval(sys.argv[6])
    # else:
    #     other_params = {}

    exp_name = "strategy_discovery"
    model_index = 3318 #models = [3315, 3316, 3317, 3318, 3323, 3324, 3325, 3326]
    optimization_criterion = "likelihood"
    pid = 172
    other_params = {"plotting": False}

    if exp_name != "strategy_discovery":
        number_of_trials = 35
    else:
        number_of_trials = 120

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


    if "exp_attributes" not in other_params:
        exp_attributes = {
            "exclude_trials": None,  # Trials to be excluded
            "block": "training",  # Block of the experiment
            "experiment": None,  # Experiment object can be passed directly with
            # Experiment object can be passed directly with pipeline and normalized features attached
            "click_cost": click_cost
        }
        other_params["exp_attributes"] = exp_attributes

    if optimization_criterion == "likelihood":
        num_sim = 1
    else:
        num_sim = 10

    if "optimization_params" not in other_params:
        optimization_params = {
            "optimizer": "hyperopt",
            "num_simulations": num_sim,
            "max_evals": 2,
            "click_cost": click_cost
        }
        other_params["optimization_params"] = optimization_params

    r_data, sim_data = fit_model(
        exp_name=exp_name,
        pid=pid,
        number_of_trials=number_of_trials,
        model_index=model_index,
        optimization_criterion=optimization_criterion,
        data_path=f"results/cm/{exp_name}",
        **other_params,
    )

    # print(r_data)
    # print(sim_data)
