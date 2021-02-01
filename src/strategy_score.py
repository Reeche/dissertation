import sys
import numpy as np
from collections import defaultdict
from src.utils.learning_utils import pickle_load, pickle_save, create_dir
from src.env.generic_mouselab import GenericMouselabEnv
from src.utils.planning_strategies import strategy_dict

strategy = int(sys.argv[1])
exp_num = sys.argv[2]
num_simulations = int(sys.argv[3])

exp_pipelines = pickle_load("data/exp_pipelines.pkl")
cluster_map = pickle_load("data/non_problematic_clusters.pkl")
exp_nums = ["v1.0", "c1.1", "c2.1_dec", "T1.1"]
strategy_scores = defaultdict(lambda: defaultdict(int))
scores = []
gts = []
for _ in range(num_simulations):
    pipeline = exp_pipelines[exp_num]
    env = GenericMouselabEnv(num_trials=1, pipeline=pipeline)
    gts.append(tuple(env.ground_truth[0]))
    clicks = strategy_dict[strategy+1](env.present_trial)
    score = env.present_trial.node_map[0].calculate_max_expected_return()
    scores.append(score)
print(len(set(gts)))
print(np.mean(scores))
d = "results/strategy_scores"
create_dir(d)
pickle_save(scores, f"{d}/{exp_num}_{strategy}.pkl")

