import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv(f"../../../Desktop/CogSci analysis/matlab_bic_learning_86.csv")
# df = pd.read_csv(f"results_2000_iterations/matlab_bic_planning_87.csv")
# df = pd.read_csv(f"results_2000_iterations/matlab_bic_threecond_87.csv")
# del df[df.columns[0]]
# df = df.drop(df.index[1]) # drop row with pid
#
# model_list = [27, 31, 59, 63, 91, 95, 123, 127, 155, 159, 187, 191, 411,
#               415, 443, 447, 475, 479, 507, 511, 539, 543, 571, 575, 603,
#               607, 635, 639, 667, 671, 699, 703, 731, 735, 763, 767, 987,
#               991, 1019, 1023, 1051, 1055, 1083, 1087, 1115, 1119, 1147,
#               1151, 1179, 1183, 1211, 1215, 1243, 1247, 1275, 1279, 1307,
#               1311, 1339, 1343, 1563, 1567, 1595, 1599, 1627, 1631, 1659,
#               1663, 1691, 1695, 1723, 1727, 1755, 1759, 1819, 1823, 1851,
#               1855, 1915, 1918, 1919, 1947, 1951, 2011, 2015, 5134]
# df.columns= model_list
#
# bic_diff = df[[1855, 1919]].copy()
# bic_diff["diff"] = bic_diff[1855] - bic_diff[1919]
#
# # count for how many of those are +3 and how many are -3
# print("larger than 3: ", sum(i > 3.2 for i in bic_diff["diff"]))
# print("less than -3: ", sum(i < -3.2 for i in bic_diff["diff"]))

# plot histogram of difference
# n, bins, patches = plt.hist(bic_diff["diff"], 50, density=True, facecolor='g', alpha=0.75)
# plt.savefig("bichistogram.png")
# plt.show()
# plt.close()


### chi^2 test of frequencies based on BMS results_2000_iterations
"""
three cond: 
        exp_r: [0.4358 0.5642]
           xp: [0.0715 0.9285]

planning amount: 
        exp_r: [0.3775 0.6225]
           xp: [0.0232 0.9768]

"""

from scipy.stats import chisquare
n_exp1 = 146
n_exp2 = 78
p_avg = 0.42
chi_results = chisquare([0.4358*n_exp1, 0.5642*n_exp1, 0.3775*n_exp2, 0.6225*n_exp2],
                        f_exp=[p_avg*n_exp1, (1-p_avg)*n_exp1, p_avg*n_exp2, (1-p_avg)*n_exp2])
print(chi_results)