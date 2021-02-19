import sys
from collections import defaultdict

import numpy as np

from ..utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions
from ..env.generic_mouselab import GenericMouselabEnv
from ..utils.planning_strategies import strategy_dict

"""
This file allows you to simulate the strategies given the environment and condition
v1.0: increasing_variance
c1.1: constant_variance
c2.1_dec: decreasing_variance

How to run: python3 calculate_strategy_score.py <strategy> <exp_num> <num_simulation>
Example: python3 calculate_strategy_score.py 1 c1.1 100
"""

# strategy = int(sys.argv[1])
# exp_num = sys.argv[2]
# num_simulations = int(sys.argv[3])
exp_num = "c1.1"
num_simulations = 50000 #use 100k

score_list = {}
# loops through all strategies and saves into a list
for strategy in range(0, 89):
    exp_pipelines = learning_utils.pickle_load("data/exp_pipelines.pkl")
    cluster_map = learning_utils.pickle_load("data/non_problematic_clusters.pkl")
    exp_nums = ["v1.0", "c1.1", "c2.1_dec", "T1.1"]
    strategy_scores = defaultdict(lambda: defaultdict(int))
    scores = []
    gts = []
    for _ in range(num_simulations):
        pipeline = exp_pipelines[exp_num]
        env = GenericMouselabEnv(num_trials=1, pipeline=pipeline)
        gts.append(tuple(env.ground_truth[0]))
        clicks = strategy_dict[strategy + 1](env.present_trial)
        #score = env.present_trial.node_map[0].calculate_max_expected_return()
        score = env.present_trial.node_map[0].calculate_max_expected_return() - (len(clicks)-1)
        scores.append(score)
    # print(len(set(gts)))
    # print(np.mean(scores))
    # d = "results/strategy_scores"
    # create_dir(d)
    # pickle_save(scores, f"{d}/{exp_num}_{strategy}.pkl")

    score_list.update({strategy: np.mean(scores)})
print(exp_num)
print(dict(sorted(score_list.items(), key=lambda item: item[1], reverse=True)))
