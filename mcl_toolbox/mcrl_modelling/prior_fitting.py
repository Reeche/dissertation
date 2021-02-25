import os
import sys

import numpy as np
import pandas as pd

path_to_toolbox = os.path.join("/home/vfelso","Rationality-Enhancement-Toolbox/")
sys.path.append(path_to_toolbox)

from condor_utils import create_sub_file, submit_sub_file

model_nums = [1729,1825,1921] # Model numbers
exp_nums = ["IRL1"]
optimization_criterion = "pseudo_likelihood" #choose one of ["pseudo_likelihood", "mer_performance_error", "performance_error"]

# range(155):#180 for v1.0
pid_range = np.unique(pd.read_csv("../data/human/IRL1/mouselab-mdp.csv")["pid"])
print(pid_range)
for exp_num in exp_nums:
    for pid in pid_range:
            for model_index in model_nums:
                args = [exp_num, model_index, optimization_criterion, pid]
                sub_file = create_sub_file("fit_mcrl_models.py", args,  py_version = '3.8', logs = False, errs=False, outputs = False, process_arg = False, num_runs = 1,
                                            num_cpus=1)
                submit_sub_file(sub_file, remove_on_submission=True, bid=2)

