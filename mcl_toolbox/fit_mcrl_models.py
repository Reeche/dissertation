import sys
import os
import random
import numpy as np
from pathlib import Path
from mcl_toolbox.global_vars import structure, model, strategies, features
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import (
    pickle_save,
    get_normalized_features,
    Participant,
    create_dir,
    get_number_of_actions_from_branching,
    construct_repeated_pipeline,
    construct_reward_function
)
from mcl_toolbox.utils.utils import get_all_pid_for_env
from mcl_toolbox.mcrl_modelling.optimizer import ParameterOptimizer

"""
Run this using: python3 fit_mcrl_models.py <exp_name> <model_index> <optimization_criterion> <pid> <string of other parameters>
<optimization_criterion> can be ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]
Example: python3 fit_mcrl_models.py v1.0 1 pseudo_likelihood 1 "{\"plotting\":True, \"optimization_params\" : {\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"
python3 fit_mcrl_models.py high_variance_high_cost 1 pseudo_likelihood 1 True hyperopt 2 2
Use the code in mcrl_modelling/prior_fitting.py to submit jobs to the cluster.
"""


# todo: not all optimization methods with all models. Need to add this somewhere in the code


def prior_fit(
    exp_name,
    model_index,
    optimization_criterion,
    pid,
    plotting=False,
    optimization_params={
        "optimizer": "hyperopt",
        "num_simulations": 5,
        "max_evals": 200,
    },
    **kwargs,
):
    """

    :param exp_name: experiment name, which is the folder name of the experiment in ../data/human/
    :param model_index: model index, as displayed in rl_models.csv
    :param optimization_criterion: as string, choose one of ["pseudo_likelihood", "mer_performance_error", "performance_error"]
    :param pid: participant id, as int
    :param optimization_params: parameters for ParameterOptimizer.optimize, passed in as a dict
    :return:
    """

    # create directory to save priors in
    parent_directory = Path(__file__).parents[1]
    prior_directory = os.path.join(
        parent_directory, f"results/mcrl/{exp_name}/{exp_name}_priors"
    )
    create_dir(prior_directory)
    # and directory to save fit model info in
    model_info_directory = os.path.join(
        parent_directory, f"results/mcrl/{exp_name}/info_{exp_name}_data"
    )
    create_dir(model_info_directory)

    # create directory to save the reward/mers data of participant (mers) and algorithm (reward)
    reward_info_directory = os.path.join(
        parent_directory, f"results/mcrl/{exp_name}/reward_{exp_name}_data"
    )
    create_dir(reward_info_directory)

    # add directory for reward plots, if plotting
    if plotting:
        plot_directory = os.path.join(
            parent_directory, f"results/mcrl/plots/{exp_name}_plots"
        )
        create_dir(plot_directory)

    num_trials = 35
    # load experiment specific info
    # For the new experiment that are not either v1.0, c1.1, c2.1_dec, F1 or IRL1
    if exp_num not in ["v1.0", "c1.1", "c2.1_dec"]:
        reward_dist = "categorical"
        reward_structure = exp_num
        reward_distributions = construct_reward_function(
            structure.reward_levels[reward_structure], reward_dist
        )
        repeated_pipeline = construct_repeated_pipeline(
            structure.branchings[exp_num], reward_distributions, num_trials
        )
        exp_pipelines = {exp_num: repeated_pipeline}
    else:
        # list of all experiments, e.g. v1.0, T1.1 only has the transfer after training (20 trials)
        exp_pipelines = structure.exp_pipelines
        if exp_num not in structure.exp_reward_structures:
            raise (ValueError, "Reward structure not found.")
        reward_structure = structure.exp_reward_structures[exp_num]

    if exp_num not in exp_pipelines:
        raise (ValueError, "Experiment pipeline not found.")
    pipeline = exp_pipelines[exp_num]  # select from exp_pipeline the selected v1.0
    # pipeline is a list of len 30, each containing a tuple of 2 {[3, 1, 2], some reward function}
    pipeline = [pipeline[0] for _ in range(100)]

    normalized_features = get_normalized_features(
        reward_structure)  # tuple of 2

    branching = structure.branchings[exp_name]
    excluded_trials = structure.excluded_trials[exp_name]

    # load model specific info
    learner_attributes = model.model_attributes.iloc[model_index].to_dict()
    learner = learner_attributes["model"]

    # load strategy space based on model parameters
    strategy_space_type = learner_attributes["strategy_space_type"]  # in the csv
    strategy_space_type = strategy_space_type if strategy_space_type else "microscope"
    strategy_space = strategies.strategy_spaces[strategy_space_type]

    # prepare participant and env for the optimization
    participant = Participant(
        exp_name, pid, excluded_trials=excluded_trials, get_strategies=False
    )
    participant.first_trial_data = participant.get_first_trial_data()
    participant.all_trials_data = participant.get_all_trials_data()
    # print(len(participant.all_trials_data["actions"]))
    env = GenericMouselabEnv(
        len(participant.envs), pipeline=pipeline, ground_truth=participant.envs
    )

    # TODO document why
    if learner == "rssl":
        num_priors = 2 * len(strategy_space)
    else:
        if learner_attributes["habitual_features"] == "habitual":
            num_priors = len(features.implemented)
        else:
            num_priors = len(features.microscope)

    # TODO document why
    use_pseudo_rewards = learner_attributes["use_pseudo_rewards"]
    pr_weight = learner_attributes["pr_weight"]
    if not pr_weight:
        learner_attributes["pr_weight"] = 1
    if not use_pseudo_rewards and pr_weight:
        raise AssertionError("Not sure why this is not allowed #TODO")

    # prepare learner_attributes
    num_actions = get_number_of_actions_from_branching(
        branching
    )  # Find out number of actions
    if learner_attributes["habitual_features"] == "habitual":
        curr_features = features.implemented
    else:
        curr_features = features.microscope

    learner_attributes = dict(
        features=curr_features,
        normalized_features=normalized_features,
        num_priors=num_priors,
        strategy_space=strategy_space,
        no_term=not learner_attributes["term"],
        num_actions=num_actions,
        **learner_attributes,
    )
    del learner_attributes["term"]  # TODO why do we delete "term" and put in "no_term"

    # optimization
    optimizer = ParameterOptimizer(
        learner, learner_attributes, participant, env
    )  # learner is the model chosen
    res, prior, obj_fn = optimizer.optimize(
        optimization_criterion, **optimization_params
    )
    # print(res[0])  # prior information
    losses = [trial["result"]["loss"] for trial in res[1]]
    print(f"Loss: {min(losses)}")
    min_index = np.argmin(losses)
    if plotting:
        reward_data = optimizer.plot_rewards(
            i=min_index,
            path=os.path.join(
                plot_directory, f"{pid}_{optimization_criterion}_{model_index}.png"
            ),
            plot=True,
        )
    # save the reward data
    pickle_save(
        reward_data,
        os.path.join(
            reward_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"
        ),
    )

    # save priors
    pickle_save(
        (res, prior),
        os.path.join(
            prior_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"
        ),
    )

    # Run simulations given prior and other fitted parameters, num of simulations: how many runs
    (r_data, sim_data), p_data = optimizer.run_hp_model(
        res[0], optimization_criterion, num_simulations=1
    )
    # print(sim_data['info'], len(sim_data['info']))
    # info of simulated data
    pickle_save(
        sim_data,
        os.path.join(
            model_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"
        ),
    )


if __name__ == "__main__":
    random.seed(123)
    # exp_num = sys.argv[1]
    # model_index = int(sys.argv[2])
    # optimization_criterion = sys.argv[3]
    # pid = int(sys.argv[4])
    # plotting = sys.argv[5]
    # optimization_params = {
    #     "optimizer": str(sys.argv[6]),
    #     "num_simulations": int(sys.argv[7]),
    #     "max_evals": int(sys.argv[8]),
    # }

    exp_num = "high_variance_high_cost"
    pid_list = get_all_pid_for_env(exp_num)
    print(pid_list)
    model_index = 1918
    optimization_criterion = "pseudo_likelihood"
    plotting = True
    pid = 1
    optimization_params = {'optimizer': "hyperopt", 'num_simulations': 2, 'max_evals': 2} #30; 400
    # for pid in pid_list:
    prior_fit(
        exp_num, model_index, optimization_criterion, pid, plotting, optimization_params
    )
