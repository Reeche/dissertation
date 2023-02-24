import pandas as pd

from condor_utils import submit_sub_file

bid = 1
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.

exp_num = 'v1.0'
# missing_df = pd.read_csv(f'../missing_{exp_num}.csv')

# add missing pid + model
# 1000
# data = [[29, 256], [1, 276], [38, 280], [5, 284], [6, 288], [43, 2047],
#         [73, 2064], [6, 2069], [29, 2074], [5, 2079], [5, 2084]]
#400
data = [[6, 276], [1, 284]]

missing_df = pd.DataFrame(data, columns=["pid", "model"])

for index, row in missing_df.iterrows():
    with open("parameters.txt", "a") as parameters:
        # need first model and then pid
        args = [exp_num, row[1], 'likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)