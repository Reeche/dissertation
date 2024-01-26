import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

# object2 = pd.read_pickle("../../results/cm/inferred_strategies/strategy_discovery_training_test/strategies.pkl")
# object = pd.read_pickle("../../results/inferred_participant_sequences/strategy_discovery/1_training_temperature.pkl")
# object3 = pd.read_pickle("../../results/cm/inferred_strategies/strategy_discovery_training/strategies.pkl")
# object4 = pd.read_pickle("../../results/cm/inferred_strategies/strategy_discovery_training/temperatures.pkl")
# object5 = pd.read_pickle("../../results/cm/inferred_strategies/v1.0_training/temperatures.pkl")
object = pd.read_pickle("../../results_mb_2000_v2/mcrl/strategy_discovery_mb/38_likelihood_full.pkl")
print(object)
