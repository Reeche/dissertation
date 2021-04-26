import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results2/mcrl/v1.0/reward_v1.0_data/173_pseudo_likelihood_861.pkl") #small num sim: 1
print(len(object))
object = pd.read_pickle("../../results2/mcrl/v1.0/reward_v1.0_data/1_pseudo_likelihood_861.pkl") #large num sim: 15
print(len(object))
#print(object[1])

object2 = pd.read_pickle("../../results/mcrl/v1.0/reward_v1.0_data/173_pseudo_likelihood_861.pkl")
print(len(object2))
object2 = pd.read_pickle("../../results/mcrl/v1.0/reward_v1.0_data/1_pseudo_likelihood_861.pkl")
print(len(object2))
#print(object2.shape)

