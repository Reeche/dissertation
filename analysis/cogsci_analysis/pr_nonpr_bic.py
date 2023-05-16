import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv(f"../../../Desktop/CogSci analysis/matlab_bic_learning_86.csv")
# df = pd.read_csv(f"results_2000_iterations/matlab_bic_planning_87.csv")
# df = pd.read_csv(f"results_2000_iterations/matlab_bic_threecond_87.csv")
# del df[df.columns[0]]
# df = df.drop(df.index[1])  # drop row with pid
#
# model_list = [13,
#               15,
#               29,
#               31,
#               34,
#               35,
#               38,
#               39,
#               42,
#               43,
#               46,
#               47,
#               109,
#               111,
#               125,
#               127,
#               130,
#               131,
#               134,
#               135,
#               138,
#               139,
#               142,
#               143,
#               157,
#               159,
#               173,
#               175,
#               178,
#               179,
#               182,
#               183,
#               186,
#               187,
#               190,
#               191,
#               253,
#               255,
#               269,
#               271,
#               274,
#               275,
#               278,
#               279,
#               282,
#               283,
#               286,
#               287,
#               301,
#               303,
#               317,
#               319,
#               322,
#               323,
#               326,
#               327,
#               330,
#               331,
#               334,
#               335,
#               397,
#               399,
#               413,
#               415,
#               418,
#               419,
#               422,
#               423,
#               426,
#               427,
#               430,
#               431,
#               445,
#               447,
#               477,
#               479,
#               482,
#               483,
#               490,
#               1743,
#               491,
#               494,
#               495,
#               502,
#               503,
#               1756]
#
# df.columns = model_list
#
# bic_diff = df[[1855, 1919]].copy()
# bic_diff["diff"] = bic_diff[1855] - bic_diff[1919]
#
# # count for how many of those are +3 and how many are -3
# print("larger than 3: ", sum(i > 3.2 for i in bic_diff["diff"]))
# print("less than -3: ", sum(i < -3.2 for i in bic_diff["diff"]))

##plot histogram of difference
# n, bins, patches = plt.hist(bic_diff["diff"], 50, density=True, facecolor='g', alpha=0.75)
# plt.savefig("bichistogram.png")
# plt.show()
# plt.close()


### chi^2 test of frequencies based on BMS
# """
# three cond:
#         exp_r: [0.4358 0.5642]
#            xp: [0.0715 0.9285]
#
# planning amount:
#         exp_r: [0.3775 0.6225]
#            xp: [0.0232 0.9768]
#
# """
#
from scipy.stats import chisquare
## pr
n_exp1 = 146
n_exp2 = 78
# p_avg = 0.39 #propotion explained by PR for both experiments
# chi_results = chisquare([0.42153*n_exp1, 0.57847*n_exp1, 0.37029*n_exp2, 0.62971*n_exp2],
#                         f_exp=[p_avg*n_exp1, (1-p_avg)*n_exp1, p_avg*n_exp2, (1-p_avg)*n_exp2])
# print(chi_results)

## td
p_avg = 0.39 #propotion explained by PR for both experiments
chi_results = chisquare([0.35145*n_exp1, 0.64855*n_exp1, 0.30981*n_exp2, 0.69019*n_exp2],
                        f_exp=[p_avg*n_exp1, (1-p_avg)*n_exp1, p_avg*n_exp2, (1-p_avg)*n_exp2])
print(chi_results)