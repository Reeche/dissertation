import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

# object = pd.read_pickle("../../results/mcrl/low_variance_low_cost/click_low_variance_low_cost_data/3_number_of_clicks_likelihood_1823.pkl")
# object2 = pd.read_pickle("../../results/mcrl/low_variance_low_cost/reward_low_variance_low_cost_data/3_number_of_clicks_likelihood_1823.pkl")
# object3 = pd.read_pickle("../../results/mcrl/low_variance_low_cost/info_low_variance_low_cost_data/3_number_of_clicks_likelihood_1823.pkl")
# object4 = pd.read_pickle("../../results/mcrl/low_variance_low_cost/low_variance_low_cost_priors/3_number_of_clicks_likelihood_95.pkl")

object = pd.read_pickle("../data/normalized_values/high_variance_low_cost/max.pkl")
print(len(object))
