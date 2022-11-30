import pickle
import sys

from mcl_toolbox.global_vars import pickle_load, structure
from mcl_toolbox.utils import learning_utils

exp_num = sys.argv[1]
pid = sys.argv[2]
model_id = sys.argv[3]
opt_crit = sys.argv[4]


results_data = pickle_load("../results/mcrl/{}_data/{}_{}_1.pkl".format(
    exp_num, pid, model_id
))
print(results_data)

results_priors = pickle_load("../results/mcrl/{}_priors/{}_{}_{}.pkl".format(
    exp_num, pid, opt_crit, model_id
))
print('\n')
print(results_priors[0])
