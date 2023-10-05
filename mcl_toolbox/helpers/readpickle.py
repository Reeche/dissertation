import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

# object = pd.read_pickle("../../results/mcrl/high_variance_low_cost_priors/4_likelihood_489.pkl")
object2 = pd.read_pickle(f"../../results_vanilla_models/mcrl/low_variance_low_cost_mb/3_likelihood.pkl")

print(object)
