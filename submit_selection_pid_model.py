import pandas as pd
import sys
from condor_utils import submit_sub_file

bid = 25

exp_num = 'strategy_discovery'
# recovered_model_type = "non_learning", "habitual", "mf", "hybrid"]
recovered_model_type = "non_learning"

data = pd.read_csv(f'missing_{recovered_model_type}.csv')
# data = [[147, 491], [28, 491], [368, 491]]

if exp_num == 'strategy_discovery':
    num_trial = 120
else:
    num_trial = 35

missing_df = pd.DataFrame(data, columns=["pid", "model_index"])

for index, row in missing_df.iterrows():
    with open("parameters.txt", "a") as parameters:
        args = [exp_num, row[1], 'likelihood', row[0], num_trial, recovered_model_type]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)


# for index, row in missing_df.iterrows():
#     with open("parameters.txt", "w") as parameters:
#         if exp_num == 'strategy_discovery':
#             num_trial = 120
#         else:
#             num_trial = 35
#         args = [exp_num, row[1], 'likelihood', row[0], num_trial]
#         args_str = " ".join(str(x) for x in args) + "\n"
#         parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
