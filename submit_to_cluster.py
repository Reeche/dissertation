import os
import sys
import time
from condor_utils import create_sub_file, submit_sub_file

bid = 10
script = 'mcl_toolbox/fit_mcrl_models.py'  # The file that you want to run on the cluster.

# exp_num = ['high_variance_high_cost']
# models = ['1823', '1919']
# pid_dict = {'high_variance_high_cost': [1]}

exp_num = ['high_variance_high_cost', 'high_variance_low_cost', 'low_variance_high_cost', 'low_variance_low_cost']
#models = ['31', '63', '95', '127', '159', '191', '607', '639', '671', '703', '735', '767',
#          '1183', '1215', '1247', '1279', '1311', '1343', '1759', '1855']

models = ['1823', '1919', '415', '447', '479', '511', '991', '1023', '1055', '1087']


pid_dict = {
    'high_variance_high_cost': [0, 1, 10, 18, 22, 25, 30, 32, 38, 41, 46, 47, 49, 57, 60, 63, 65, 70, 74, 76, 81, 83,
                                88, 89, 94, 103, 108, 109, 111, 114, 116, 118, 125, 129, 134, 139, 149, 150, 156, 161,
                                164, 167, 169, 173, 177, 182, 188, 191, 195, 198, 199, 204],
    'high_variance_low_cost': [4, 7, 8, 17, 20, 23, 26, 29, 33, 35, 40, 48, 50, 51, 53, 56, 58, 64, 71, 78, 82, 87, 92,
                               93, 95, 96, 101, 112, 117, 119, 122, 126, 131, 133, 136, 141, 145, 146, 151, 154, 158,
                               162, 168, 175, 180, 185, 187, 189, 193, 197, 202, 205],
    'low_variance_high_cost': [2, 13, 14, 16, 21, 24, 28, 31, 36, 37, 43, 45, 54, 61, 62, 68, 69, 73, 79, 80, 84, 86,
                               90, 97, 98, 100, 102, 107, 120, 124, 128, 132, 135, 138, 140, 144, 147, 153, 157, 160,
                               163, 166, 171, 174, 181, 183, 192, 194, 201, 203, 206],
    'low_variance_low_cost': [3, 5, 6, 9, 11, 12, 15, 19, 27, 34, 39, 42, 44, 52, 55, 59, 66, 67, 72, 75, 77, 85, 91,
                              99, 104, 105, 106, 110, 113, 115, 121, 123, 127, 130, 137, 142, 143, 148, 152, 155, 159,
                              165, 170, 172, 176, 178, 179, 184, 186, 190, 196, 200, 207]}


num_simulation = 30
max_eval = 400
for exp_num_ in exp_num:
    for models_ in models:
        pids = pid_dict.get(exp_num_)
        if pids:
            for pid in pids:
                args = [exp_num_, models_, 'pseudo_likelihood', pid, True, 'hyperopt', num_simulation, max_eval]
                sub_file = create_sub_file(script, args, process_arg=False, num_runs=1, num_cpus=1, req_mem=2000,
                                           logs=True,
                                           errs=True, outputs=True)
                submit_sub_file(sub_file, bid)
                time.sleep(1)

# python3 mcl_toolbox/fit_mcrl_models.py v1.0 861 pseudo_likelihood 1 hyperopt 2 2
# python3 mcl_toolbox/fit_mcrl_models.py <exp_num> <model_index> <optimization criterion> <pid> hyperopt <num_simulation> <max_eval>