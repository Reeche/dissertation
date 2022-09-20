import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/mcrl/low_variance_low_cost_priors/3_likelihood_1919.pkl")
object2 = pd.read_pickle("../../results/mcrl/low_variance_low_cost_data/3_1919_1.pkl")


print(len(object))
