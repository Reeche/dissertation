import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/mcrl_testing/high_variance_high_cost_priors/18_likelihood_1823.pkl")
# object3 = pd.read_pickle("../../results/mcrl/c1.1_priors/2_likelihood_1823.pkl")
# object2 = pd.read_pickle("../../results/mcrl/c1.1_data/2_1412_1.pkl")


print(len(object))
