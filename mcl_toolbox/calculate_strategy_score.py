import sys
from collections import defaultdict

import numpy as np

from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.planning_strategies import strategy_dict

"""
This file allows you to simulate the strategies given the environment and condition
v1.0: increasing_variance
c1.1: constant_variance
c2.1_dec: decreasing_variance

How to run: python3 calculate_strategy_score.py <strategy> <exp_num> <num_simulation>
Example: python3 calculate_strategy_score.py c2.1_dec 100000
"""

exp_num = sys.argv[1]
num_simulations = int(sys.argv[2])

# exp_num = "v1.0"
# num_simulations = 200000  # 200k seems to be stable, therefore 250k
exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")
score_list = {}

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
        clicks = strategy_dict[strategy + 1](env.present_trial)
        score = env.present_trial.node_map[0].calculate_max_expected_return() - (len(clicks) - 1)
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
dir = "../results/strategy_scores"
learning_utils.create_dir(dir)
learning_utils.pickle_save(score_results, f"{dir}/{exp_num}_strategy_scores.pkl")

# todo: add cluster scores