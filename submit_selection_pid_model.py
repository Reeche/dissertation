import pandas as pd
import sys
from condor_utils import submit_sub_file

bid = 50
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.

# exp_num = 'high_variance_high_cost'
exp_num = sys.argv[1]
missing_df = pd.read_csv(f'missing_{exp_num}.csv')

#
# # low_variance_low_cost_data
# data = [[60, 1756]]
# missing_df = pd.DataFrame(data, columns=["pid", "model"])

for index, row in missing_df.iterrows():
    with open("parameters.txt", "a") as parameters:
        # need first model and then pid
        if exp_num in ['v1.0', 'c2.1', 'c1.1']:
            args = [exp_num, row[1], 'pseudo_likelihood', row[0], 35]
        else:
            args = [exp_num, row[1], 'number_of_clicks_likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
