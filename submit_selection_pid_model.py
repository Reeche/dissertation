import pandas as pd
import sys
from condor_utils import submit_sub_file

bid = 25
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.

# exp_num = 'v1.0'
exp_num = sys.argv[1]
# missing_df = pd.read_csv(f'missing_{exp_num}.csv')

# add missing pid + model
# v1.0
# data = [[18, 38], [18, 182], [18, 190], [21, 38], [21, 279], [24, 134], [24, 238], [24, 278],
#         [45, 86], [45, 286], [56, 279], [68, 38], [104, 239], [124, 38], [124, 182],
#         [132, 182], [140, 191], [140, 278], [144, 47], [144, 190], [160, 238], [165, 190]]

# c1.1
# data = [[37, 191], [37, 286], [120, 94], [120, 190], [135, 87], [135, 239], [135, 286],
#         [143, 134], [143, 182], [143, 190], [143, 286], [151, 39], [151, 182], [151, 190],
#         [151, 286], [157, 135], [157, 286], [168, 38]]

# c2.1
# data = [[16, 38], [16, 39], [16, 95], [26, 278], [31, 47], [33, 231], [72, 135], [72, 286], [79, 431], [99, 46],
#         [115, 287], [118, 191], [130, 87], [134, 230], [134, 286], [134, 287], [145, 231], [162, 286], [172, 287]]

# high_variance_low_cost_data
# data = [[8, 423], [53, 135], [119, 46]]

# high_variance_high_cost_data
# data = [[1, 422], [25, 142], [38, 47], [38, 430], [47, 46], [47, 334], [47, 335], [47, 383], [47, 422], [49, 94],
#         [49, 382], [76, 38], [76, 374], [83, 86], [83, 95], [83, 143], [83, 279], [134, 95], [134, 382], [195, 47],
#         [195, 375]]
#
# # low_variance_low_cost_data
data = [[27, 38], [27, 39], [27, 46], [27, 134], [27, 286], [52, 86], [99, 238], [123, 182], [123, 278], [142, 38],
        [142, 46], [159, 86], [207, 239]]
missing_df = pd.DataFrame(data, columns=["pid", "model"])

for index, row in missing_df.iterrows():
    with open("parameters.txt", "a") as parameters:
        # need first model and then pid
        args = [exp_num, row[1], 'likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
