import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

object = pd.read_pickle("exp_pipelines.pkl")
print(len(object))
print(object)

# kl_cluster_map
# cluster_scores
