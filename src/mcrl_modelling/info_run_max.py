import sys
from optimizer import ParameterOptimizer
from src.utils.learning_utils import pickle_load, pickle_save, \
    get_normalized_features, Participant, create_dir
from src.env.generic_mouselab import GenericMouselabEnv
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger()
#logger.setLevel(logging.CRITICAL)
strategy_spaces = {'participant': [6, 11, 14, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 37, 39, 40, 42, 43, 44, 50, 56, 57, 58,
                                    63, 64, 65, 67, 70, 76, 79, 87, 88],
                  'microscope': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 84, 85, 86, 87, 88, 89]}

def expand_path(path, v, s):
    if v:
        path+= s
    return path

control_pids = [1, 2, 6, 9, 11, 14, 18, 21, 24, 27, 37, 38, 44, 50, 55, 56, 58, 66, 76, 79, 85, 89, 90, 98, 99,
                100, 104, 111, 113, 118, 119, 123, 126, 129, 139, 142, 144, 153, 154]

def main():
    exp_pipelines = pickle_load("../data/exp_pipelines.pkl")
    exp_reward_structures = pickle_load("../data/exp_reward_structures.pkl")
    features = pickle_load(f"../data/implemented_features.pkl")

    exp_num = sys.argv[1]
    normalized_features = get_normalized_features(exp_reward_structures[exp_num])
    pipeline = exp_pipelines[exp_num]
    num_trials = 30

    model_attributes = pd.read_csv("../models/rl_models.csv")
    model_attributes = model_attributes.where(pd.notnull(model_attributes), None)

    pid = int(sys.argv[4])
    optimization_criterion = sys.argv[3]
    
    model_index = int(sys.argv[2])
    num_models = len(model_attributes)
    rewards = []

    num_simulations = 5
    num_evals = 50 # For hyperopt only
    excluded_trials = None
    if exp_num in ["c1.1"]:
        excluded_trials = list(range(30))
    participant = Participant(exp_num, pid, excluded_trials=excluded_trials,
                            get_strategies=False)
    participant_trials = True
    if participant_trials:
        env = GenericMouselabEnv(len(participant.envs), pipeline=pipeline,
                                    ground_truth=participant.envs)
    else:
        num_trials = 100
        env = GenericMouselabEnv(num_trials, pipeline=[pipeline[0]]*num_trials)

    models = []
    attributes = []
    model_indices = [model_index]
    d = f"results/info_{exp_num}_max"
    d2 = f"results/info_{exp_num}_data_max"
    create_dir(d)
    create_dir(d2)
    for model_index in model_indices:
        print(f"::::::::::::::Model Number {model_index}:::::::::::::::")
        learner_attributes = model_attributes.iloc[model_index].to_dict()
        learner = learner_attributes['model']
        print(learner, learner_attributes)

        num_actions = 13 # Find out number of actions
        strategy_space_type = learner_attributes['strategy_space_type']
        strategy_space_type = strategy_space_type if strategy_space_type else 'microscope'
        strategy_space = strategy_spaces[strategy_space_type]

        if learner == "rssl":
            num_priors = 2*len(strategy_space)
        else:
            num_priors = len(features)

        use_pseudo_rewards = learner_attributes['use_pseudo_rewards']
        pr_weight = learner_attributes['pr_weight']
        if not pr_weight:
            learner_attributes['pr_weight'] = 1
        if not use_pseudo_rewards and pr_weight:
            continue

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
        losses = [trial['result']['loss'] for trial in res[1]]
        min_index = np.argmin(losses)

        num_trials = 1000
        pipeline=[pipeline[0]]*num_trials
        from src.env.generic_mouselab import DummyParticipantNew
        participant = DummyParticipantNew(pipeline, num_trials)
        env = GenericMouselabEnv(num_trials, pipeline)
        optimizer = ParameterOptimizer(learner, learner_attributes, participant, env)
        (r_data, sim_data), p_data = optimizer.run_hp_model_nop(res[0], "reward",
                                                        num_simulations=10)
        pickle_save(sim_data, f"{d2}/{pid}_{optimization_criterion}_{model_index}.pkl")

if __name__ == "__main__":
    main()
