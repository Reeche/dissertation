import sys

import pandas as pd
from generic_mouselab import GenericMouselabEnv
from learning_utils import pickle_load, pickle_save, \
    get_normalized_features, Participant, create_dir
from optimizer import ParameterOptimizer


def main():
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = pickle_load("data/exp_reward_structures.pkl")
    features = pickle_load(f"data/implemented_features.pkl")
    model_attributes = pd.read_csv("rl_models.csv", index_col=0)
    model_attributes = model_attributes.where(pd.notnull(model_attributes), None)

    exp_num = sys.argv[1]
    model_index = int(sys.argv[2])
    optimization_criterion = sys.argv[3]
    pid = int(sys.argv[4])

    normalized_features = get_normalized_features(exp_reward_structures[exp_num])
    pipeline = exp_pipelines[exp_num]
    pipeline = construct_repeated_pipeline(branching, reward_function, num_simulations)

    num_simulations = 5
    num_evals = 200 # For hyperopt only
    excluded_trials = None
    if exp_num in ["c1.1"]:
        excluded_trials = list(range(30))

    participant = Participant(exp_num, pid, excluded_trials=excluded_trials,
                            get_strategies=False)
    env = GenericMouselabEnv(len(participant.envs), pipeline=pipeline,
                                    ground_truth=participant.envs)

    models = []
    attributes = []

    d = f"results/{exp_num}_priors"
    create_dir(d)

    print(f"::::::::::::::Model Number {model_index}:::::::::::::::")
    learner_attributes = model_attributes.iloc[model_index].to_dict()
    learner = learner_attributes['model']
    if learner == "rssl":
        num_priors = 2*len(strategy_space)
    else:
        num_priors = len(features)

    num_actions = 13 # Find out number of actions
    strategy_space_type = learner_attributes['strategy_space_type']
    strategy_space_type = strategy_space_type if strategy_space_type else 'microscope'
    strategy_space = strategy_spaces[strategy_space_type]

    use_pseudo_rewards = learner_attributes['use_pseudo_rewards']
    pr_weight = learner_attributes['pr_weight']
    if not pr_weight:
        learner_attributes['pr_weight'] = 1
    if not use_pseudo_rewards and pr_weight:
        raise AssertionError("Not sure why this is not allowed #TODO")

    participant.first_trial_data = participant.get_first_trial_data()
    participant.all_trials_data = participant.get_all_trials_data()

    learner_attributes = dict(features=features, normalized_features=normalized_features,
            num_priors=num_priors,
            strategy_space=strategy_space,
            no_term = not learner_attributes['term'],
            num_actions = num_actions,
            **learner_attributes)
    del learner_attributes['term']
    models.append(learner)
    attributes.append(learner_attributes)
    optimizer = ParameterOptimizer(learner, learner_attributes, participant, env)
    res, prior, obj_fn = optimizer.optimize(optimization_criterion, num_simulations=num_simulations,
                optimizer="hyperopt",
                        max_evals=num_evals)
    print(res, prior)

    pickle_save((res, prior), f"{d}/{pid}_{optimization_criterion}_{model_index}.pkl")

if __name__ == "__main__":
    main()
