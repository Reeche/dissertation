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

How to run: python3 calculate_strategy_score.py <strategy> <exp_num> <num_simulation>
Example: python3 mcl_toolbox/calculate_strategy_score.py c2.1_dec 1000 1
"""

exp_num = sys.argv[1]
num_simulations = int(sys.argv[2])  # at least 200k is recommended
click_cost = float(sys.argv[3])
if len(sys.argv) > 4:
    reward_level = sys.argv[4]

score_list = {}
click_list = {}

# if you are using v1.0, c1.1, c2.1_dec or T1, you can uncomment this line
# exp_pipelines = learning_utils.pickle_load("mcl_toolbox/data/exp_pipelines.pkl")
exp_pipelines = structure.exp_pipelines

# Adjust the environment that you want to simulate in global_vars.py
# reward_dist = "categorical"
# num_trials = 35
# reward_distributions = learning_utils.construct_reward_function(
#     structure.reward_levels[reward_level], reward_dist
# )
# repeated_pipeline = learning_utils.construct_repeated_pipeline(
#     structure.branchings[exp_num], reward_distributions, num_trials
# )
# exp_pipelines = {exp_num: repeated_pipeline}

# loops through all strategies and saves into a list
for strategy in range(0, 89):
    print("strategy", strategy)
    strategy_scores = defaultdict(lambda: defaultdict(int))
    scores = []
    gts = []
    number_of_clicks = []
    for _ in range(num_simulations):
        pipeline = exp_pipelines[exp_num]
        env = GenericMouselabEnv(num_trials=1, pipeline=pipeline)
        gts.append(tuple(env.ground_truth[0]))
        clicks = strategy_dict[strategy + 1](
            env.present_trial
        )  # gets the click sequence
        number_of_clicks.append(len(clicks))
        receive_reward = np.random.choice([0, 1], p=[1 - click_cost, click_cost])
        score = (
            env.present_trial.node_map[0].calculate_max_expected_return() * receive_reward
            - (len(clicks) - 1) * click_cost
        )  # len(clicks) is always 13
        scores.append(score)

    print("Score", np.mean(scores))
    print("Clicks", np.mean(number_of_clicks))

    score_list.update({strategy: np.mean(scores)})
    click_list.update({strategy: np.mean(number_of_clicks)})

score_results = dict(sorted(score_list.items(), key=lambda item: item[1], reverse=True))
print("Score results", score_results)
dir = "results/cm/strategy_scores/"
learning_utils.create_dir(dir)
learning_utils.pickle_save(
    score_results, f"{dir}/{exp_num}_clickcost_{click_cost}_strategy_scores.pkl"
)
print("Number of clicks", click_list)
learning_utils.pickle_save(
    click_list, f"{dir}/{exp_num}_strategy_scores.pkl"
)
