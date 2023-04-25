import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

# object = pd.read_pickle("../../results_2000_iterations/mcrl/v1.0_priors/1_likelihood_300.pkl")
# object = pd.read_pickle("../../results_2000_iterations/mcrl/v1.0_priors/1_likelihood_5134.pkl")
# object3 = pd.read_pickle("../../results_2000_iterations/mcrl/c1.1_priors/2_likelihood_1823.pkl")
object = pd.read_pickle("../../results/cm/strategy_scores/threecond/c2.1_dec_clickcost_3_strategy_scores.pkl")
object1 = pd.read_pickle("../../results/cm/strategy_scores/threecond/c2.1_dec_clickcost_6_strategy_scores.pkl")
object2 = pd.read_pickle("../../results/cm/strategy_scores/threecond/c2.1_dec_clickcost_9_strategy_scores.pkl")


print(object)
