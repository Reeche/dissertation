import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("../../results/cm/strategy_scores/low_variance_low_cost_strategy_scores.pkl")
object2 = pd.read_pickle("../../results/cm/strategy_scores/low_variance_low_cost_numberclicks.pkl")


print(len(object))
