import sys
from collections import defaultdict

import numpy as np
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.planning_strategies import strategy_dict
from mcl_toolbox.global_vars import structure

from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

"""
This file allows you to simulate the strategies given the environment and condition
v1.0: increasing_variance
c1.1: constant_variance
c2.1_dec: decreasing_variance

"""

# exp_num = sys.argv[1]
# num_simulations = int(sys.argv[2]) # at least 200k is recommended
# click_cost = int(sys.argv[3])

score_list = {}

# Adjust the environment that you want to simulate in global_vars.py
reward_level = "adjusted_constant"
exp_num = "PL1"
reward_dist = "categorical"
num_trials = 35
num_simulations = 200000
click_cost = 1
reward_distributions = learning_utils.construct_reward_function(
    structure.reward_levels[reward_level], reward_dist
)
repeated_pipeline = learning_utils.construct_repeated_pipeline(
    structure.branchings[exp_num], reward_distributions, num_trials
)
exp_pipelines = {exp_num: repeated_pipeline}

# if you are using v1.0, c1.1, c2.1_dec or T1, you can uncomment this line
# exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")

# loops through all strategies and saves into a list
print(exp_num, num_simulations)
# todo: change this one to not hard coded
for strategy in range(0, 89):
    print("strategy", strategy)
    strategy_scores = defaultdict(lambda: defaultdict(int))
    scores = []
    gts = []
    for _ in range(num_simulations):
        pipeline = exp_pipelines[exp_num]
        env = GenericMouselabEnv(num_trials=1, pipeline=pipeline)
        gts.append(tuple(env.ground_truth[0]))
        clicks = strategy_dict[strategy + 1](
            env.present_trial
        )  # gets the click sequence
        score = (
            env.present_trial.node_map[0].calculate_max_expected_return()
            - (len(clicks) - 1) * click_cost
        )  # len(clicks) is always 13
        scores.append(score)
    # print(len(set(gts)))
    # print(np.mean(scores))
    # This saves the scores individually for each strategy
    # d = "../results/strategy_scores"
    # learning_utils.create_dir(d)
    # learning_utils.pickle_save(scores, f"{d}/{exp_num}_{strategy}.pkl")

    score_list.update({strategy: np.mean(scores)})

score_results = dict(sorted(score_list.items(), key=lambda item: item[1], reverse=True))
print(score_results)
dir = "../results/cm/strategy_scores/planningclicks/highdiscrepancy_cost1/"
learning_utils.create_dir(dir)
learning_utils.pickle_save(score_results, f"{dir}/{exp_num}_strategy_scores.pkl")
