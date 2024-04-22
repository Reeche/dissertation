import pandas as pd
import sys
from condor_utils import submit_sub_file

bid = 25
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.


# exp_num = sys.argv[1]
# missing_df = pd.read_csv(f'missing_{exp_num}.csv')


exp_num = 'strategy_discovery'
data = [[45, 479]]


missing_df = pd.DataFrame(data, columns=["pid", "model"])

for index, row in missing_df.iterrows():
    with open("parameters.txt", "a") as parameters:
        args = [exp_num, row[1], 'likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
