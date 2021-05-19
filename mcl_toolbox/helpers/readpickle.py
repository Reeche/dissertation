import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

# object = pd.read_pickle("../../results/mcrl/c1.1/info_c1.1_data/81_pseudo_likelihood_605.pkl")
# object2 = pd.read_pickle("../../results/mcrl/v1.0/v1.0_priors/17_pseudo_likelihood_1757.pkl")
#
# object = pd.read_pickle("../data/implemented_features.pkl")
# object2 = pd.read_pickle("../data/microscope_features.pkl")
# object = pd.read_pickle("../../results/cm/strategy_scores/c1.1_strategy_scores.pkl")
object = pd.read_pickle("../data/strategy_space.pkl")
print(len(object))




