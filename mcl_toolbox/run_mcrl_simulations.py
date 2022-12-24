import ast
import copy
import sys
import os
import time
from pathlib import Path

from mcl_toolbox.utils.learning_utils import create_dir
from mcl_toolbox.utils.model_utils import ModelFitter

from mcl_toolbox.global_vars import pickle_load

"""
Run this using:
python3 fit_mcrl_models.py <exp_name> <model_index> <optimization_criterion> <pid> <number_of_trials> <string of other parameters>
<optimization_criterion> can be ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]
Example: python3 mcl_toolbox/fit_mcrl_models.py v1.0 1 pseudo_likelihood 1 35 "{\"plotting\":True, \"sim_params\" :{\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"
"""


# todo: not all optimization methods work with all models. Need to add this somewhere in the code
# Likelihood computation is currently available for all reinforce and lvoc models
#
# TODO: Give right strategy weights to RSSL model for likelihood computation


def run_simulations(
    exp_name,
    pid,
    number_of_trials,
    model_index,
    optimization_criterion,
    exp_attributes=None,
    sim_params=None,
    data_path=None,
    save_path=None
):

    # create directory to save priors in
    if save_path is None:
        save_path = Path(__file__).resolve().parents[1].joinpath("results/mcrl")
    else:
        save_path.mkdir(parents=True, exist_ok=True)

    prior_dir = save_path.joinpath(f"{exp_name}_priors")


    model_info_directory = save_path.joinpath(f"{exp_name}_data")

    mf = ModelFitter(
        exp_name,
        exp_attributes=exp_attributes,
        data_path=data_path,
        number_of_trials=number_of_trials,
    )

    file_extension = ""
    if "learn_from_actions" in sim_params:
        file_extension += str(sim_params["learn_from_actions"])
    else:
        file_extension += "1"

    if "learn_from_unrewarded" in sim_params:
        file_extension += str(int(sim_params["learn_from_unrewarded"]))
    else:
        file_extension += "0"

    prior_file = os.path.join(
                    prior_dir, f"{pid}_{optimization_criterion}_{model_index}_{file_extension}.pkl"
                )

    print(prior_file)

    results_priors = pickle_load(prior_file)

    best_params = results_priors[0][0]
    print(best_params)

    mf.simulate_params(
        model_index,
        best_params,
        pid=pid,
        sim_dir=model_info_directory,
        sim_params=sim_params,
    )

if __name__ == "__main__":
    exp_name = sys.argv[1]
    model_index = int(sys.argv[2])
    optimization_criterion = sys.argv[3]
    pid = sys.argv[4]
    number_of_trials = int(sys.argv[5])
    # other_params = {"plotting": True}
    other_params = {}
    if len(sys.argv) > 6:
        other_params = ast.literal_eval(sys.argv[6])
    else:
        other_params = {}

    # exp_name = "v1.0"
    # model_index = 1919 #6527
    # optimization_criterion = "likelihood"
    # # optimization_criterion = "pseudo_likelihood"
    # pid = 6  # 1, 5, 6, 10, 15
    # other_params = {"plotting": True}
    # number_of_trials = 35

    click_cost = 0.25 if exp_name == "scarcity_scarce" else 1

    if "exp_attributes" not in other_params:
        exp_attributes = {
            "exclude_trials": None,  # Trials to be excluded
            "block": None,  # Block of the experiment
            "experiment": None,
            # Experiment object can be passed directly with pipeline and normalized features attached
            "click_cost": click_cost,
        }
        other_params["exp_attributes"] = exp_attributes

    if "sim_params" not in other_params:
        optimization_params = {
            "optimizer": "hyperopt",
            "num_simulations": 30,
            "max_evals": 400,  # 400 - number of param updates
        }

        other_params["sim_params"] = optimization_params

    for attribute, default_val in zip(
            ["optimizer", "num_simulations", "max_evals"],
            ["hyperopt", 30, 400]
    ):
        if attribute not in other_params["sim_params"]:
            other_params["sim_params"][attribute] = default_val
    if "pct_rewarded" not in other_params:
        other_params["sim_params"]["pct_rewarded"] = 1.0
    else:
        other_params["sim_params"]["pct_rewarded"] = other_params["pct_rewarded"]

    tic = time.perf_counter()
    run_simulations(
        exp_name=exp_name,
        pid=pid,
        number_of_trials=number_of_trials,
        model_index=model_index,
        optimization_criterion=optimization_criterion,
        **other_params,
    )
    toc = time.perf_counter()
    print(f"Took {toc - tic:0.4f} seconds")
