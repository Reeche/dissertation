import pandas as pd
import numpy as np
from learning_utils import pickle_load, pickle_save, create_dir

exp_name = "v1.0"
blocks = {"IRL1":{"test":30} , "v1.0":{"test":20}} #block name and expected number of trials
model_indices = [1729,1825,1921]
opt_criteria = ["pseudo_likelihood"]

create_dir("results/priors")

mouselab_data = pd.read_csv("../data/human/{}/mouselab-mdp.csv".format(exp_name))
blocks_groupby = mouselab_data[mouselab_data["block"].isin(blocks)].groupby(["pid", "block"]).count()
block_counts = blocks_groupby.pivot_table(values="trial_id", columns=["block"], index="pid")

valid_indices = (block_counts.apply(lambda row : np.all([row[key]==val for key,val in blocks[exp_name].items()]),axis=1))
valid_pids = valid_indices[valid_indices].index.values

for opt_criterion in opt_criteria:
    for model in model_indices:
        all_priors = {}
        for pid in valid_pids:
            curr_pid_prior = pickle_load("results/{}_priors/{}_{}_{}.pkl".format(exp_name, pid, opt_criterion, model))
            all_priors[pid] = curr_pid_prior[0]
        pickle_save(all_priors, "results/priors/{}_{}_{}.pkl".format(exp_name, opt_criterion, model))
