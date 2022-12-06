import pandas as pd

from condor_utils import submit_sub_file

bid = 1
script = 'fit_mcrl_models.py'  # The file that you want to run on the cluster.

exp_num = 'v1.0'
# missing_df = pd.read_csv(f'../missing_{exp_num}.csv')

# add missing pid + model
# high variance high cost
# data = [[0, 1]]

# low variance high cost
# data = [[13, 1358], [13, 1382], [16, 1405], [21, 1345], [21, 1354], [31, 1192], [31, 1355],
#         [43, 1351], [80, 1388], [80, 1397], [100, 1355], [157, 1458], [160, 1357], [163, 1351],
#         [201, 1393]]

# c1.1
# data = [[100, 1]]

# c2.1
# data = [[0, 1], [13, 1483], [20, 1893], [103, 81], [103, 1692], [108, 1233], [113, 311],
#         [113, 529], [115, 875], [128, 946], [128, 1551], [130, 1530], [142, 1855], [149, 982],
#         [149, 987], [164, 842]]

# v1.0
# data = [[6, 1782], [66, 1801], [68, 1970], [68, 1977], [68, 2000], [75, 870],
#         [77, 1687], [80, 146], [82, 1366], [82, 1390], [94, 992], [98, 433]]

data = [[68, 1980]]
missing_df = pd.DataFrame(data, columns=["pid", "model"])

for index, row in missing_df.iterrows():
    #todo: first time replace but then append, currently, need to delete the previous one manually
    with open("parameters.txt", "a") as parameters:
        # need first model and then pid
        args = [exp_num, row[1], 'likelihood', row[0], 35]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
