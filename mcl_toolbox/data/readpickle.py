import pandas as pd
import sys
from mcl_toolbox.utils import learning_utils, distributions
import pickle

sys.modules["learning_utils"] = learning_utils
sys.modules["distributions"] = distributions

all = pd.read_pickle("implemented_features.pkl") #all features
microscope = pd.read_pickle("microscope_features.pkl") #non habitual
non_learning = pd.read_pickle("non_learning_features.pkl") #non learning
mf = pd.read_pickle("model_free_features.pkl") #non learning
# set_difference = set(all) - set(microscope)
# print(object)

##remove feature 49 'trial_level_std' from microscope_features.pkl
##Create a copy of the original list
mb = microscope.copy()

# Remove the specific item "a" from the new list
mb.remove("max_uncertainty")
mb.remove("uncertainty")
mb.remove("successor_uncertainty")
mb.remove("trial_level_std")
mb.remove("get_level_observed_std")

# save as new pickle file
with open('model_free_features.pkl', 'wb') as file:
    pickle.dump(mb, file)

