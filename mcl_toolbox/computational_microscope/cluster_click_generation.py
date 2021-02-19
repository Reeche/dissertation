#import os
import sys
from condor_utils import create_sub_file, submit_sub_file

exp_nums = ["v1.0", "c1.1_old", "c2.1_dec", "T1.1"]
print(exp_nums)
num_simulations = 10000
for exp_num in exp_nums:
    args = [exp_num, num_simulations]
    sub_file = create_sub_file("gen_clicks.py", args, process_arg = True, num_runs = 90,
                                num_cpus=1)
    submit_sub_file(sub_file, 31)