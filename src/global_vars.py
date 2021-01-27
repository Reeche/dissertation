import numpy as np
import seaborn as sns
from learning_utils import pickle_load


class structure:
    branchings = {"v1.0": [3, 1, 2], "F1": [3, 1, 2], "T1.1": [3, 1, 1, 2, 3], 'c1.1': [3, 1, 2], 'c2.1': [3, 1, 2]}
    level_values = [[0], [-4, -2, 2, 4], [-8, -4, 4, 8], [-48, -24, 24, 48]]
    reward_levels = {'high_increasing': level_values[1:], 'high_decreasing': level_values[1:][::-1],
                     'low_constant': const_var_values * 3, 'large_increasing': list(zip(np.zeros(5), [1, 2, 4, 8, 32]))}

    reward_type = {'F1': 'categorical', 'c1.1': 'categorical', 'c2.1': 'categorical', 'T1.1': 'normal',
                   'v1.0': 'categorical'}

    small_level_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3,
                       5: 1, 6: 2, 7: 3, 8: 3, 9: 1, 10: 2, 11: 3, 12: 3}
    const_var_values = [[-10, -5, 5, 10]]

class strategies:
    num_strategies = 89
    # strategy_space = list(range(1, num_strategies + 1))
    # problematic_strategies = [19, 20, 25, 35, 38, 52, 68, 77, 81, 83] #the microscope strategies are obtained from this
    strategy_spaces = {'participant': [6, 11, 14, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 37, 39, 40, 42, 43, 44, 50, 56, 57, 58,
                                    63, 64, 65, 67, 70, 76, 79, 87, 88],
                  'microscope': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
                                33, 34, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 84, 85, 86, 87, 88, 89]}

    strategy_weights = pickle_load("data/microscope_weights.pkl")
    strategy_distances = pickle_load("data/L2_distances.pkl")

class features:
    features = pickle_load("data/microscope_features.pkl")

class hierarchical_params:
    # normalize = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 72.0, 2.0, 2.0] #unused, but was at top of file
    precision_epsilon = 1e-4
    # quadrature_max_degree = 1e5 #unused, but was at top of file

class misc:
    control_pids = [1, 2, 6, 9, 11, 14, 18, 21, 24, 27, 37, 38, 44, 50, 55, 56, 58, 66, 76, 79, 85, 89, 90, 98, 99,
                    100, 104, 111, 113, 118, 119, 123, 126, 129, 139, 142, 144, 153, 154]
    machine_eps = np.finfo(float).eps #machine epsilon


class plotting:
    sns.set_style('whitegrid')