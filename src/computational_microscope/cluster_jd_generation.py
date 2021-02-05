#import os
import sys
from condor_utils import create_sub_file, submit_sub_file

exp_nums = ["v1.0", "c1.1_old", "c2.1_dec", "T1.1"]
num_simulations = 10000
for exp_num in exp_nums:
    for s_s in range(90):
        args = [exp_num, s_s, num_simulations]
        sub_file = create_sub_file("compute_jd.py", args, process_arg = True, num_runs = 90,
                                    num_cpus=1)
        submit_sub_file(sub_file, 31)