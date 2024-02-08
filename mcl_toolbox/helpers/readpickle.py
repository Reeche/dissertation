import pandas as pd
# import sys
# from mcl_toolbox.utils import learning_utils, distributions

# sys.modules["learning_utils"] = learning_utils
# sys.modules["distributions"] = distributions

# object1 = pd.read_pickle(f"../../final_results/pure/high_variance_high_cost_priors/0_likelihood_491.pkl")
# object2 = pd.read_pickle(f"../../final_results/pure/high_variance_high_cost_priors/0_likelihood_1743.pkl")
# object3 = pd.read_pickle(f"../../final_results/pure/non_learning/high_variance_high_cost_priors/0_likelihood_1756.pkl")

object = pd.read_pickle("../../results_mb_2000_v3/mcrl/strategy_discovery_mb/2_likelihood_full.pkl")

print(object)
