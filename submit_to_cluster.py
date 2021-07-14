import os
import sys
import time
from condor_utils import create_sub_file, submit_sub_file

bid = 500
script = 'mcl_toolbox/fit_mcrl_models.py'  # The file that you want to run on the cluster.

exp_num = ['v1.0']
models = ['5134']
pid_dict = {'v1.0': [21]}

# exp_num = ['v1.0', 'c2.1_dec', 'c1.1']
# models = ['5134']

# pid_dict = {
#     'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77,
#              80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
#              148, 150, 154, 155, 158, 160, 165, 169, 173],
#     'c2.1_dec': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
#                  79,
#                  84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142,
#                  145, 149, 152, 156, 162, 164, 166, 170, 172],
#     'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
#              83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
#              151, 153, 157, 159, 161, 163, 167, 168, 171]}

# first and fourth run. 30, 100
# second run: 10; 400
# third run: 30; 400
# fifth run: 30; 600

num_simulation = 5
max_eval = 50
for exp_num_ in exp_num:
    for models_ in models:
        pids = pid_dict.get(exp_num_)
        if pids:
            for pid in pids:
                args = [exp_num_, models_, 'pseudo_likelihood', pid, 'hyperopt', num_simulation, max_eval]
                sub_file = create_sub_file(script, args, process_arg=False, num_runs=1, num_cpus=1, req_mem=2000,
                                           logs=True,
                                           errs=True, outputs=True)
                submit_sub_file(sub_file, bid)
                time.sleep(1)


# python3 mcl_toolbox/fit_mcrl_models.py v1.0 861 pseudo_likelihood 1 hyperopt 2 2
# python3 mcl_toolbox/fit_mcrl_models.py <exp_num> <model_index> <optimization criterion> <pid> hyperopt <num_simulation> <max_eval>
