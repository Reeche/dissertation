import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/mcrl/v1.0_model_based/data/51_likelihood.pkl")
# object2 = pd.read_pickle(f"../../results/mcrl/v1.0_priors/1_likelihood_300.pkl")

print(object)
