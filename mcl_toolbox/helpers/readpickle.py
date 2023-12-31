import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/cm/inferred_strategies/stroop_training/strategies.pkl")
# object = pd.read_pickle("../../results_mb_8000_v2/mcrl/v1.0_mb/1_likelihood_full.pkl")

print(object)
