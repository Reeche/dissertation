import pickle

from mcl_toolbox.global_vars import pickle_load, structure
from mcl_toolbox.utils import learning_utils
strategies = pickle_load("../results/inferred_participant_sequences/scarcity_scarce/631a24dfca851aac1863e4a4_begin_strategies.pkl")
print(strategies)
