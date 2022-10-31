import pandas as pd

from condor_utils import submit_sub_file

bid = 1
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.

exp_num = 'v1.0'
missing_df = pd.read_csv(f'../missing_{exp_num}.csv')
for index, row in missing_df.iterrows():
    #todo: first time replace but then append, currently, need to delete the previous one manually
    with open("parameters.txt", "a") as parameters:
        args = [exp_num, row[1], 'likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
