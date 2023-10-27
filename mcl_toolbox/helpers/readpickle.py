import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/cm/inferred_strategies/strategy_discovery_training/strategies.pkl")
# object2 = pd.read_pickle("../../strategy_discovery_results/mcrl/strategy_discovery_priors/10_likelihood_491.pkl")
# object = pd.read_pickle("../../results_mb_2000/mcrl/strategy_discovery_mb/5_likelihood.pkl")

print(object)
