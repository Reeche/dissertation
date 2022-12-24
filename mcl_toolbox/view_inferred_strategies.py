import pickle
import sys

from mcl_toolbox.global_vars import pickle_load, structure
from mcl_toolbox.utils import learning_utils

exp_num = sys.argv[1]
pid = sys.argv[2]
block = sys.argv[3]


strategies = pickle_load("../results/inferred_participant_sequences/{}/{}_{}_strategies.pkl".format(
    exp_num, pid, block
))
print(strategies)
