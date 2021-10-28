import unittest

from mouselab.envs.registry import registry
from parameterized import parameterized

from pathlib import Path
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import create_mcrl_reward_distribution, construct_repeated_pipeline, get_normalized_features

from itertools import product

"""
Currently tests to make sure we can instantiate a ModelFitter object in all project
AND that we can fit all optimization criteria
python3 -m unittest tests.test_model_fitter
"""

# here, the name is just used for loading in participant data
new_pipeline = [["F1", "high_increasing", 3, 31],
                    ["c1.1", "low_constant", 1, 15]]
old_pipeline = [["F1", "test", 3],
                ["c1.1","train_final", 1]]

metrics = ["likelihood", "pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]

class TestModelFitter(unittest.TestCase):
    @parameterized.expand([[*param, metric] for param, metric in product(new_pipeline, metrics)])
    def test_new_pipeline_with_exp(self, exp_name, exp_setting, pid, num_trials, metric):
        # For IRL project
        reward_distributions = create_mcrl_reward_distribution(exp_setting)
        branching = registry(exp_setting).branching
        pipeline = construct_repeated_pipeline(branching, reward_distributions, num_trials)

        experiment = Experiment(exp_name, pids=[pid], data_path=Path(__file__).parents[1].joinpath("data"))
        experiment.attach_pipeline(pipeline)
        experiment.normalized_features = get_normalized_features(exp_setting)
        # test model fitting
        mf = ModelFitter(exp_name, exp_attributes={"experiment": experiment})

        res, prior, _ = mf.fit_model(
            1729,
            pid,
            metric,
            {
                "optimizer"      : "hyperopt",
                "num_simulations": 2,
                "max_evals"      : 2,
            },
            params_dir=None,
        )
        # reached end
        self.assertTrue(True)

    @parameterized.expand([[*param, metric] for param, metric in product(new_pipeline, metrics)])
    def test_new_pipeline_with_kwargs(self, exp_name, exp_setting, pid, num_trials, metric):
        # For new MCRL project
        mf = ModelFitter(exp_name, exp_setting=exp_setting, num_trials=num_trials)
        # test model fitting
        res, prior, _ = mf.fit_model(
            1729,
            pid,
            metric,
            {
                "optimizer"      : "hyperopt",
                "num_simulations": 2,
                "max_evals"      : 2,
            },
            params_dir=None,
        )
        # reached end
        self.assertTrue(True)
    @parameterized.expand([[*param, metric] for param, metric in product(old_pipeline, metrics)])
    def test_old_pipeline_with_exp(self, exp_name, block, pid, metric):
        # For old MCRL project
        mf = ModelFitter(exp_name, exp_attributes={"block": block}, data_path=Path(__file__).parents[1].joinpath("data"))
        # test model fitting
        res, prior, _ = mf.fit_model(
            1729,
            pid,
            metric,
            {
                "optimizer"      : "hyperopt",
                "num_simulations": 2,
                "max_evals"      : 2,
            },
            params_dir=None,
        )
        # reached end
        self.assertTrue(True)
