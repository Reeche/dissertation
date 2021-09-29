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
        file_path  -- Location of the file to be loaded, as pathlib object.
    Returns:
        Unpickled object
    """
    with open(str(file_path), "rb") as file_obj:
        unpickled_obj = RenameUnpickler(file_obj).load()
    return unpickled_obj


class structure:
    excluded_trials = {
        "v1.0": None,
        "F1": None,
        "T1.1": list(range(11)),
        "c1.1": list(range(30)),
        "c2.1": None,
        "IRL1": list(range(30, 66)),
    }

    branchings = {
        "v1.0": [3, 1, 2],
        "F1": [3, 1, 2],
        "T1.1": [3, 1, 1, 2, 3],
        "c1.1": [3, 1, 2],
        "c2.1": [3, 1, 2],
        "IRL1": [3, 1, 2],
    }
    level_values = [[0], [-4, -2, 2, 4], [-8, -4, 4, 8], [-48, -24, 24, 48]]
    const_var_values = [[-10, -5, 5, 10]]
    reward_levels = {
        "high_increasing": level_values[1:],
        "high_decreasing": level_values[1:][::-1],
        "low_constant": const_var_values * 3,
        "large_increasing": list(zip(np.zeros(5), [1, 2, 4, 8, 32])),
    }

    reward_exps = {
        "F1": "categorical",
        "c1.1": "categorical",
        "c2.1": "categorical",
        "T1.1": "normal",
        "v1.0": "categorical",
        "IRL1": "categorical",
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
    exp_pipelines = pickle_load(file_location.joinpath("data/exp_pipelines.pkl"))
    # for some reason F1 is missing one trial

    exp_pipelines["F1"].append(exp_pipelines["F1"][0])

    # this maps experiment code to the text version of its reward level, e.g. 'low_constant' or 'large_increasing', before was pickle_load("data/exp_reward_structures.pkl")
    exp_reward_structures = {
        "v1.0": "high_increasing",
        "F1": "high_increasing",
        "c1.1": "low_constant",
        "c2.1_dec": "high_decreasing",
        "T1.1": "large_increasing",
        "IRL1": "high_increasing",
    }

    normalized_value_directories = {
        "increasing_variance": "high_increasing",
        "constant_variance": "low_constant",
        "decreasing_variance": "high_decreasing",
        "transfer_task": "large_increasing",
    }


class model:
    model_attributes = pd.read_csv(
        os.path.join(file_location, "models/rl_models.csv"), index_col=0
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
        "participant": [
            6,
            11,
            14,
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
            37,
            39,
            40,
            42,
            43,
            44,
            50,
            56,
            57,
            58,
            63,
            64,
            65,
            67,
            70,
            76,
            79,
            87,
            88,
        ],
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
    microscope = pickle_load(
        file_location.joinpath("data/microscope_features.pkl")
    )  # this is 51 features
    implemented = pickle_load(
        file_location.joinpath("data/implemented_features.pkl")
    )  # this is 56 features


class hierarchical_params:
    # normalize = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 72.0, 2.0, 2.0] #unused, but was at top of file
    precision_epsilon = 1e-4
    # quadrature_max_degree = 1e5 #unused, but was at top of file


class misc:
    control_pids = [
        1,
        2,
        6,
        9,
        11,
        14,
        18,
        21,
        24,
        27,
        37,
        38,
        44,
        50,
        55,
        56,
        58,
        66,
        76,
        79,
        85,
        89,
        90,
        98,
        99,
        100,
        104,
        111,
        113,
        118,
        119,
        123,
        126,
        129,
        139,
        142,
        144,
        153,
        154,
    ]


class plotting:
    sns.set_style("whitegrid")
