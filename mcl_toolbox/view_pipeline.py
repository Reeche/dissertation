import pickle

from mcl_toolbox.global_vars import pickle_load, structure
from mcl_toolbox.utils import learning_utils
pipelines = pickle_load("data/exp_pipelines.pkl")
print(len(pipelines["v1.0"]))
