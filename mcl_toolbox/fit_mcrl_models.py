import random

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.global_vars import *
from mcl_toolbox.mcrl_modelling.optimizer import ParameterOptimizer
from mcl_toolbox.utils.learning_utils import (
    pickle_save,
    get_normalized_features,
    Participant,
    create_dir,
    get_number_of_actions_from_branching,
)

"""
Run this using: python3 fit_mcrl_models.py <exp_name> <model_index> <optimization_criterion> <pid> <string of other parameters>
<optimization_criterion> can be ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]
Example: python3 fit_mcrl_models.py v1.0 1 pseudo_likelihood 1 "{\"plotting\":True, \"optimization_params\" : {\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"

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
    prior_directory = os.path.join(parent_directory, f"results/mcrl/{exp_name}_priors")
    create_dir(prior_directory)
    # and directory to save fit model info in
    model_info_directory = os.path.join(
        parent_directory, f"results/mcrl/info_{exp_name}_data"
    )
    create_dir(model_info_directory)

    # add directory for reward plots, if plotting
    if plotting:
        plot_directory = os.path.join(
            parent_directory, f"results/mcrl/plots/{exp_name}_plots"
        )
        create_dir(plot_directory)

    # load experiment specific info
    normalized_features = get_normalized_features(
        structure.exp_reward_structures[exp_name]
    )
    pipeline = structure.exp_pipelines[exp_name]
    branching = structure.branchings[exp_name]
    excluded_trials = structure.excluded_trials[exp_name]

    # load model specific info
    learner_attributes = model.model_attributes.iloc[model_index].to_dict()
    learner = learner_attributes["model"]

    # load strategy space based on model parameters
    strategy_space_type = learner_attributes["strategy_space_type"]
    strategy_space_type = strategy_space_type if strategy_space_type else "microscope"
    strategy_space = strategies.strategy_spaces[strategy_space_type]

    # prepare participant and env for the optimization
    participant = Participant(
        exp_name, pid, excluded_trials=excluded_trials, get_strategies=False
    )
    participant.first_trial_data = participant.get_first_trial_data()
    participant.all_trials_data = participant.get_all_trials_data()
    print(len(participant.all_trials_data["actions"]))
    env = GenericMouselabEnv(
        len(participant.envs), pipeline=pipeline, ground_truth=participant.envs
    )

    # TODO document why
    if learner == "rssl":
        num_priors = 2 * len(strategy_space)
    else:
        num_priors = len(features.implemented)

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
    optimizer = ParameterOptimizer(learner, learner_attributes, participant, env)
    res, prior, obj_fn = optimizer.optimize(
        optimization_criterion, **optimization_params
    )
    print(res[0])
    losses = [trial["result"]["loss"] for trial in res[1]]
    print(f"Loss: {min(losses)}")
    min_index = np.argmin(losses)
    if plotting:
        reward_data = optimizer.plot_rewards(
            i=min_index, path=os.path.join(plot_directory, f"{pid}.png")
        )
    # save priors
    pickle_save(
        (res, prior),
        os.path.join(
            prior_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"
        ),
    )

    # TODO: document what is this? Is this running simulations given priors?
    (r_data, sim_data), p_data = optimizer.run_hp_model(
        res[0], optimization_criterion, num_simulations=30
    )
    print(sim_data["info"], len(sim_data["info"]))
    pickle_save(
        sim_data,
        os.path.join(
            model_info_directory, f"{pid}_{optimization_criterion}_{model_index}.pkl"
        ),
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
    model_index = 1
    optimization_criterion = "pseudo_likelihood"
    pid = 1
    plotting = True
    other_params = {"optimizer": "hyperopt", "num_simulations": 5, "max_evals": 2}
    prior_fit(
        exp_name, model_index, optimization_criterion, pid, plotting, **other_params
    )
