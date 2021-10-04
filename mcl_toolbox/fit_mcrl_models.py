import os
import random

from mcl_toolbox.global_vars import *
from mcl_toolbox.utils.learning_utils import create_dir
from mcl_toolbox.utils.model_utils import ModelFitter

"""
Run this using: 
python3 fit_mcrl_models.py <exp_name> <model_index> <optimization_criterion> <pid> <string of other parameters>
<optimization_criterion> can be ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]
Example: python3 fit_mcrl_models.py v1.0 1 pseudo_likelihood 1 "{\"plotting\":True, \"optimization_params\" : 
{\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"

Use the code in mcrl_modelling/prior_fitting.py to submit jobs to the cluster.
"""


# todo: not all optimization methods work with all models. Need to add this somewhere in the code
# Likelihood computation is currently available for all reinforce and lvoc models
#

# TODO: Add ability to remove trials and also use arbitrary pipelines


def prior_fit(
    exp_name,
    model_index,
    optimization_criterion,
    pid,
    plotting=True,
    optimization_params=None,
    **kwargs,
):
    """

    :param exp_name: experiment name, which is the folder name of the experiment in ../data/human/
    :param model_index: model index, as displayed in rl_models.csv
    :param optimization_criterion: as string, choose one of: ["pseudo_likelihood", "mer_performance_error", "performance_error"]
    :param pid: participant id, as int
    :param optimization_params: parameters for ParameterOptimizer.optimize, passed in as a dict
    :return:
    """

    # create directory to save priors in
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(parent_directory, f"results/mcrl/{exp_name}_priors")
    create_dir(prior_directory)
    # and directory to save fit model info in
    model_info_directory = os.path.join(
        parent_directory, f"results/mcrl/info_{exp_name}_data"
    )
    create_dir(model_info_directory)

    # add directory for reward plots, if plotting
    plot_directory = None
    if plotting:
        plot_directory = os.path.join(
            parent_directory, f"results/mcrl/plots/{exp_name}_plots"
        )
        create_dir(plot_directory)

    mf = ModelFitter(exp_name)
    res, prior, obj_fn = mf.fit_model(
        model_index,
        pid,
        optimization_criterion,
        optimization_params,
        params_dir=prior_directory,
    )
    mf.simulate_params(
        model_index,
        res[0],
        pid=pid,
        sim_dir=model_info_directory,
        plot_dir=plot_directory,
    )


if __name__ == "__main__":
    random.seed(123)
    # exp_name = sys.argv[1]
    # model_index = int(sys.argv[2])
    # optimization_criterion = sys.argv[3]
    # pid = int(sys.argv[4])
    # other_params = {}
    # if len(sys.argv)>5:
    #     other_params = ast.literal_eval(sys.argv[5])

    exp_name = "v1.0"
    model_index = 1729
    optimization_criterion = "reward"
    pid = 15
    plotting = True
    optimization_params = {
        "optimizer": "hyperopt",
        "num_simulations": 1,
        "max_evals": 20,
    }
    prior_fit(
        exp_name,
        model_index,
        optimization_criterion,
        pid,
        plotting,
        optimization_params=optimization_params,
    )
