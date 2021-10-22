from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# seed random number generator
seed(1)
# prepare data
data1 = [2.970675, 4.49771, 2.75044, 3.93417, 3.941575, 3.95372, 5.624195, 5.313955, 3.0, 3.0]
data2 = [0, 1, 1, 1, 1, 2, 2, 2, 3, 3]
# calculate cohen's d
d = cohend(data1, data2)
print('Cohens d: %.3f' % d)


# parameters for power analysis
effect = d #0.8
alpha = 0.05
power = 0.8
# perform power analysis
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)