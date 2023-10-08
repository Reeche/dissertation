import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results_vanilla_models/mcrl/high_variance_high_cost_priors/0_number_of_clicks_likelihood_479.pkl")
# object2 = pd.read_pickle(f"../../results_vanilla_models/mcrl/v1.0_mb/1_likelihood.pkl")

print(object)
