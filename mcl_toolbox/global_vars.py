import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

file_location = Path(__file__).parents[0]


# RenameUnpickler from https://stackoverflow.com/a/53327348
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        # we need these modules in order to load pickled files depending on them,
        # pickled with earlier versions of the code
        if module == "learning_utils":
            renamed_module = "mcl_toolbox.utils.learning_utils"
        elif module == "distributions":
            renamed_module = "mcl_toolbox.utils.distributions"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def pickle_load(file_path):
    """
    Load the pickle file located at 'filepath'
    Params:
        file_path  -- Location of the file to be loaded.
    Returns:
        Unpickled object
    """
    if not os.path.exists(file_path):
        head, tail = os.path.split(__file__)
        if file_path[0] == "/":
            new_path = os.path.join(head, file_path[1:])
        else:
            new_path = os.path.join(head, file_path)
        if os.path.exists(new_path):
            file_path = new_path
        else:
            raise FileNotFoundError(f"{file_path} not found.")
    file_obj = open(file_path, "rb")
    return RenameUnpickler(file_obj).load()


class structure:
    # excluded trials by Yash (and Val?)
    excluded_trials = {
        "v1.0": None,
        "F1": None,
        "T1.1": list(range(11)),
        "c1.1": list(range(30)),
        "c2.1": None,
        "c2.1_dec": None,
        "low_variance_high_cost": None,
        "low_variance_low_cost": None,
        "high_variance_high_cost": None,
        "high_variance_low_cost": None,
        "strategy_discovery": None,
        "mb": None,
        "mf": None,
        "stroop": None,
    }

    branchings = {
        "v1.0": [3, 1, 2],
        "F1": [3, 1, 2],
        "T1.1": [3, 1, 1, 2, 3],
        "c1.1": [3, 1, 2],
        "c2.1": [3, 1, 2],
        "c2.1_dec": [3, 1, 2],
        "IRL1": [3, 1, 2],
        "low_variance_high_cost": [3, 1, 2],
        "low_variance_low_cost": [3, 1, 2],
        "high_variance_high_cost": [3, 1, 2],
        "high_variance_low_cost": [3, 1, 2],
        "strategy_discovery": [3, 1, 2],
        "mb": [3, 1, 2],
        "mf": [3, 1, 2],
        "stroop": [3, 1, 2],
    }
    strategy_discovery = [[0], [-1, 1], [-5], [-5, -50, 50]]
    level_values_increasing = [[0], [-4, -2, 2, 4], [-8, -4, 4, 8], [-48, -24, 24, 48]]
    # level_values_decreasing = [
    #     [0],
    #     [-6, -3, 3, 6],
    #     [-11, -6, 6, 11],
    #     [-67, -34, 34, 67],
    # ]
    # const_var_values = [[-30, -15, 15, 30]]
    const_var_values = [[-10, -5, 5, 10]]
    high_variance_values = [[-1000, -50, -50, -50, -20, -20, 50, 50, 50, 100, 100]]
    low_variance_values = [[-6, -4, -2, 2, 4, 6]]
    reward_levels = {
        "high_increasing": level_values_increasing[1:],
        "high_decreasing": level_values_increasing[1:][::-1],
        "low_constant": const_var_values * 3,
        "large_increasing": list(zip(np.zeros(5), [1, 2, 4, 8, 32])),
        "low_variance_high_cost": low_variance_values * 3,
        "low_variance_low_cost": low_variance_values * 3,
        "high_variance_high_cost": high_variance_values * 3,
        "high_variance_low_cost": high_variance_values * 3,
        "strategy_discovery": strategy_discovery * 3,
        "mb": level_values_increasing * 3,
        "mf": level_values_increasing * 3,
        "stroop": level_values_increasing * 3,
    }

    reward_exps = {
        "F1": "categorical",
        "c1.1": "categorical",
        "c2.1": "categorical",
        "c2.1_dec": "categorical",
        "T1.1": "normal",
        "v1.0": "categorical",
        "IRL1": "categorical",
        "high_variance_high_cost": "categorical",
        "high_variance_low_cost": "categorical",
        "low_variance_high_cost": "categorical",
        "low_variance_low_cost": "categorical",
        "strategy_discovery": "categorical",
        "mb": "categorical",
        "mf": "categorical",
        "stroop": "categorical",
    }

    small_level_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 1,
        6: 2,
        7: 3,
        8: 3,
        9: 1,
        10: 2,
        11: 3,
        12: 3,
    }

    """ 
    an exp pipeline is a list containing a tuple for each trial that is included, containing: 
    1) branching, e.g. [3,1,2]
    2) reward function, as a functools.partial parametrized by a function reward_function and a level distribution of a list of random variables (using construct_reward_function in learning_utils)
    the function construct_repeated_pipeline is used to create a pipeline
    """
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    # for some reason F1 is missing one trial
    exp_pipelines["F1"].append(exp_pipelines["F1"][0])

    # this maps experiment code to the text version of its reward level, e.g. 'low_constant' or 'large_increasing', before was pickle_load("data/exp_reward_structures.pkl")
    exp_reward_structures = {
        "v1.0": "high_increasing",
        "F1": "high_increasing",
        "c1.1": "low_constant",
        "c2.1": "high_decreasing",
        "c2.1_dec": "high_decreasing",
        "T1.1": "large_increasing",
        "IRL1": "high_increasing",
        "low_variance_high_cost": "low_variance_high_cost",
        "low_variance_low_cost": "low_variance_low_cost",
        "high_variance_high_cost": "high_variance_high_cost",
        "high_variance_low_cost": "high_variance_low_cost",
        "strategy_discovery": "strategy_discovery",
        "mb": "high_increasing",
        "mf": "high_increasing",
        "stroop": "high_increasing",
    }

    normalized_value_directories = {
        "increasing_variance": "high_increasing",
        "constant_variance": "low_constant",
        "decreasing_variance": "high_decreasing",
        "transfer_task": "large_increasing",
        "mb": "high_increasing",
        "mf": "high_increasing",
        "stroop": "high_increasing",
    }


class model:
    model_attributes = pd.read_csv(
        str(file_location.joinpath("models/rl_models.csv")), index_col=0
    )
    # TODO quick fix, we need to rename this column as it breaks the fit_mcrl_models.py code
    model_attributes.columns = [
        "habitual_features" if col == "features" else col
        for col in model_attributes.columns
    ]
    model_attributes = model_attributes.where(pd.notnull(model_attributes), None)


class strategies:
    num_strategies = 89
    # strategy_space = list(range(1, num_strategies + 1))
    # problematic_strategies = [19, 20, 25, 35, 38, 52, 68, 77, 81, 83] #the microscope strategies are obtained from this
    strategy_space = pickle_load(file_location.joinpath("data/strategy_space.pkl"))
    strategy_spaces = {
        "participant": [6, 11, 14, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 37, 39, 40, 42, 43, 44, 50, 56, 57, 58, 63,
            64, 65, 67, 70, 76, 79, 87, 88],
        "microscope": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            78,
            79,
            80,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
        ],
    }

    strategy_weights = pickle_load(
        file_location.joinpath("data/microscope_weights.pkl")
    )
    strategy_distances = pickle_load(file_location.joinpath("data/L2_distances.pkl"))


class features:
    hybrid_ssl_features = pickle_load(file_location.joinpath("data/hybrid_ssl_features.pkl"))  # 56 features
    model_free_habitual_features = pickle_load(
        file_location.joinpath("data/model_free_habitual_features.pkl"))  # 51 features
    non_learning_features = pickle_load(file_location.joinpath("data/non_learning_features.pkl"))  # 46 features

    # strategy discovery has 7 additional features
    sd_hybrid_ssl_features = pickle_load(
        file_location.joinpath("data/strategy_discovery_features_hybrid_ssl.pkl"))  # 63 features
    sd_model_free_habitual_features = pickle_load(
        file_location.joinpath("data/strategy_discovery_features_mf_habitual.pkl"))  # 58 features
    sd_non_learning_features = pickle_load(
        file_location.joinpath("data/strategy_discovery_features_non_learning.pkl"))  # 53 features


class hierarchical_params:
    precision_epsilon = 1e-4


class plotting:
    sns.set_style("whitegrid")


def assign_model_names(row):
    if row['class'] == 'hybrid' and row['model_index'] == "491":
        return 'hybrid Reinforce'
    elif row['class'] == 'hybrid' and row['model_index'] == "479":
        return 'hybrid LVOC'
    elif row['class'] == 'pure' and row['model_index'] == "491":
        return 'pure Reinforce'
    elif row['class'] == 'pure' and row['model_index'] == "479":
        return 'pure LVOC'
    elif row['model_index'] == "1743":
        return 'Habitual'
    elif row['model_index'] == "1756":
        return 'Non-learning'
    elif row['model_index'] == "522":
        return 'SSL'
    elif row['model_index'] == "no_assumption_level":
        return 'MB - No assumption, grouped'
    elif row['model_index'] == "no_assumption_individual":
        return 'MB - No assumption, individual'
    elif row['model_index'] == "uniform_individual":
        return 'MB - Uniform, individual'
    elif row['model_index'] == "uniform_level":
        return 'MB - Uniform, grouped'
    elif row['model_index'] == "level_level":
        return 'MB - Level, grouped'
    elif row['model_index'] == "level_individual":
        return 'MB - Level, individual'
    else:
        raise ValueError("Model class combination not found")
