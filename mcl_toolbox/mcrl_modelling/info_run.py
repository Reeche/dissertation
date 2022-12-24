import logging
import sys

import pandas as pd
from optimizer import ParameterOptimizer

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import (Participant, create_dir,
                                              get_normalized_features,
                                              pickle_load, pickle_save)

logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
strategy_spaces = {
    "participant": [
        6,
        11,
        14,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        30,
        31,
        37,
        39,
        40,
        42,
        43,
        44,
        50,
        56,
        57,
        58,
        63,
        64,
        65,
        67,
        70,
        76,
        79,
        87,
        88,
    ],
    "microscope": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        78,
        79,
        80,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
    ],
}


def expand_path(path, v, s):
    if v:
        path += s
    return path


control_pids = [
    1,
    2,
    6,
    9,
    11,
    14,
    18,
    21,
    24,
    27,
    37,
    38,
    44,
    50,
    55,
    56,
    58,
    66,
    76,
    79,
    85,
    89,
    90,
    98,
    99,
    100,
    104,
    111,
    113,
    118,
    119,
    123,
    126,
    129,
    139,
    142,
    144,
    153,
    154,
]


def main():
    exp_pipelines = pickle_load("../data/exp_pipelines.pkl")
    exp_reward_structures = pickle_load("../data/exp_reward_structures.pkl")
    features = pickle_load("../data/implemented_features.pkl")

    exp_num = sys.argv[1]
    normalized_features = get_normalized_features(exp_reward_structures[exp_num])
    pipeline = exp_pipelines[exp_num]
    # transfer_pipeline = [exp_pipelines["T1.1"][21]] + exp_pipelines["F1"][:10] + exp_pipelines["T1.1"][:20]
    # pipeline = transfer_pipeline
    num_trials = 30

    model_attributes = pd.read_csv("../models/rl_models.csv")
    model_attributes = model_attributes.where(pd.notnull(model_attributes), None)

    # pid = control_pids[int(sys.argv[1])]
    pid = int(sys.argv[4])
    optimization_criterion = sys.argv[3]

    model_index = int(sys.argv[2])

    num_simulations = 10
    num_evals = 100  # For hyperopt only
    excluded_trials = None
    if exp_num in ["c1.1"]:
        excluded_trials = list(range(30))
    participant = Participant(
        exp_num, pid, excluded_trials=excluded_trials, get_strategies=False
    )
    participant_trials = True
    if participant_trials:
        env = GenericMouselabEnv(
            len(participant.envs), pipeline=pipeline, ground_truth=participant.envs
        )
    else:
        num_trials = 100
        env = GenericMouselabEnv(num_trials, pipeline=[pipeline[0]] * num_trials)

    models = []
    attributes = []
    # for model_index in choices(list(range(num_models)), k=10):
    # model_indices = [0, 1, 64, 65, 65, 65, 1728, 1729]
    # model_indices = [0, 1, 65, 1825, 1728, 1729]
    model_indices = [model_index]
    # model_indices = [model_index]
    d = f"results/info_{exp_num}"
    # d = f"results/model_performance_error_{exp_num}"
    # d2 = f"results/model_performance_error_{exp_num}_data"
    d2 = f"results/info_{exp_num}_data"
    create_dir(d)
    create_dir(d2)
    for model_index in model_indices:
        print(f"::::::::::::::Model Number {model_index}:::::::::::::::")
        learner_attributes = model_attributes.iloc[model_index].to_dict()
        learner = learner_attributes["model"]
        print(learner, learner_attributes)

        num_actions = 13  # Find out number of actions
        strategy_space_type = learner_attributes["strategy_space_type"]
        strategy_space_type = (
            strategy_space_type if strategy_space_type else "microscope"
        )
        strategy_space = strategy_spaces[strategy_space_type]

        if learner == "rssl":
            num_priors = 2 * len(strategy_space)
        else:
            num_priors = len(features)

        use_pseudo_rewards = learner_attributes["use_pseudo_rewards"]
        pr_weight = learner_attributes["pr_weight"]
        if not use_pseudo_rewards and pr_weight:
            continue
        if not pr_weight:
            del learner_attributes["pr_weight"]

        # if "rssl" in learner or optimization_criterion in ["strategy_accuracy", "strategy_transitions"]:
        #     if strategy_space_type == "microscope":
        #         participant.strategies = pickle_load(f"results/final_strategy_inferences/{exp_num}_strategies.pkl")[pid]
        #         participant.temperature = pickle_load(f"results/final_strategy_inferences/{exp_num}_temperatures.pkl")[pid]
        #     else:
        #         participant.strategies = pickle_load(f"results/final_strategy_inferences/{exp_num}_{strategy_space_type}_strategies.pkl")[pid]
        #         participant.temperature = pickle_load(f"results/final_strategy_inferences/{exp_num}_{strategy_space_type}_temperatures.pkl")[pid]
        participant.first_trial_data = participant.get_first_trial_data()
        participant.all_trials_data = participant.get_all_trials_data()

        learner_attributes = dict(
            features=features,
            normalized_features=normalized_features,
            num_priors=num_priors,
            strategy_space=strategy_space,
            no_term=not learner_attributes["term"],
            num_actions=num_actions,
            **learner_attributes,
        )
        del learner_attributes["term"]
        models.append(learner)
        attributes.append(learner_attributes)
        optimizer = ParameterOptimizer(learner, learner_attributes, participant, env)
        res, prior, obj_fn = optimizer.optimize(
            optimization_criterion,
            num_simulations=num_simulations,
            optimizer="hyperopt",
            max_evals=num_evals,
        )
        print(res[0])
        losses = [trial["result"]["loss"] for trial in res[1]]
        print(f"Loss: {min(losses)}")
        create_dir(f"results/{exp_num}_plots")
        # reward_data = optimizer.plot_rewards(i=min_index, path=f"results/{exp_num}_plots/{pid}.png")
        # pickle_save(reward_data, f"{d}/{pid}_{optimization_criterion}_{model_index}.pkl")
        # pickle_save(res, f"{d}/{pid}_{optimization_criterion}_{model_index}.pkl")
        best_params = res[0]
        if "pr_weight" not in best_params:
            best_params["pr_weight"] = 1
        (r_data, sim_data, agent), p_data = optimizer.run_hp_model(
            best_params, optimization_criterion, num_simulations=30
        )
        # print(sim_data['info'], len(sim_data['info']))
        pickle_save(sim_data, f"{d2}/{pid}_{optimization_criterion}_{model_index}.pkl")
        # pickle_save(sim_data, f"{d2}/{pid}_{optimization_criterion}_{model_index}.pkl")
        # optimizer.plot_history(res, prior, obj_fn)
    # bms = BayesianModelSelection(models, attributes, participant, env,
    #                            optimization_criterion, num_simulations)
    # history = bms.model_selection()
    # plot_model_selection_results(history, models)


if __name__ == "__main__":
    main()
