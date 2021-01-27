import sys

from generic_mouselab import GenericMouselabEnv
from learning_utils import pickle_load, pickle_save, \
    get_normalized_features, Participant, create_dir, get_number_of_actions_from_branching
from optimizer import ParameterOptimizer
from global_vars import *

def prior_fit(exp_name, model_index, optimization_criterion, pid, optimization_params = {'optimizer':"hyperopt", 'num_simulations': 5, 'max_evals': 200}):
    '''

    :param exp_name: experiment name, which is the folder name of the experiment in ../data/human/
    :param model_index: model index, as displayed in rl_models.csv
    :param optimization_criterion: as string, choose one of ["pseudo_likelihood", "mer_performance_error", "performance_error"]
    :param pid: participant id, as int
    :param optimization_params: parameters for ParaemterOptimizer.optimize, passed in as a dict
    :return:
    '''

    # create directory to save priors in
    d = f"results/{exp_name}_priors"
    create_dir(d)

    #load experiment specific info
    normalized_features = get_normalized_features(structure.exp_reward_structures[exp_name])
    pipeline = structure.exp_pipelines[exp_name]
    branching = structure.branchings[exp_name]
    excluded_trials = structure.excluded_trials[exp_name]

    #load model specific info
    learner_attributes = model.model_attributes.iloc[model_index].to_dict()
    learner = learner_attributes['model']

    #load strategy space based on model parameters
    strategy_space_type = learner_attributes['strategy_space_type']
    strategy_space_type = strategy_space_type if strategy_space_type else 'microscope'
    strategy_space = strategies.strategy_spaces[strategy_space_type]


    #prepare participant and env for the optimization
    participant = Participant(exp_name, pid, excluded_trials=excluded_trials,
                            get_strategies=False)
    participant.first_trial_data = participant.get_first_trial_data()
    participant.all_trials_data = participant.get_all_trials_data()

    env = GenericMouselabEnv(len(participant.envs), pipeline=pipeline,
                                    ground_truth=participant.envs)

    #TODO document why
    if learner == "rssl":
        num_priors = 2*len(strategy_space)
    else:
        num_priors = len(features.implemented)

    #TODO document why
    use_pseudo_rewards = learner_attributes['use_pseudo_rewards']
    pr_weight = learner_attributes['pr_weight']
    if not pr_weight:
        learner_attributes['pr_weight'] = 1
    if not use_pseudo_rewards and pr_weight:
        raise AssertionError("Not sure why this is not allowed #TODO")

    #prepare learner_attributes
    num_actions = get_number_of_actions_from_branching(branching)  # Find out number of actions
    learner_attributes = dict(features=features.implemented, normalized_features=normalized_features,
            num_priors=num_priors,
            strategy_space=strategy_space,
            no_term = not learner_attributes['term'],
            num_actions = num_actions,
            **learner_attributes)
    del learner_attributes['term'] #TODO why do we delete "term" and put in "no_term"

    #optimization
    optimizer = ParameterOptimizer(learner, learner_attributes, participant, env)
    res, prior, obj_fn = optimizer.optimize(optimization_criterion, **optimization_params)
    print(res, prior)

    #save priors
    pickle_save((res, prior), f"{d}/{pid}_{optimization_criterion}_{model_index}.pkl")

if __name__ == "__main__":
    exp_name = sys.argv[1]
    model_index = int(sys.argv[2])
    optimization_criterion = sys.argv[3]
    pid = int(sys.argv[4])
    prior_fit(exp_name, model_index, optimization_criterion, pid)
