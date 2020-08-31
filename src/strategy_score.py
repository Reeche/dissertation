import operator
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choice
from collections import defaultdict, Counter
from analysis_utils import get_data
from learning_utils import pickle_load, pickle_save, construct_pipeline, Participant, get_normalized_features,\
                            get_normalized_feature_values, construct_reward_function, reward_levels, reward_type, \
                            construct_repeated_pipeline, create_dir, get_strategy_counts, get_cluster_dict, \
                            get_modified_weights
from sequence_utils import compute_average_click_likelihoods
from generic_mouselab import GenericMouselabEnv
from modified_mouselab import TrialSequence, reward_val, normal_reward_val, constant_reward_val, decreasing_reward_val
from planning_strategies import strategy_dict
from computational_microscope import ComputationalMicroscope
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy.linalg as LA
from scipy.special import softmax
from IPython.core.display import display, HTML
from experiment_utils import Experiment

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

