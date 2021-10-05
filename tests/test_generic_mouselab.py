import unittest

from mouselab.envs.registry import registry
from parameterized import parameterized

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import construct_repeated_pipeline, create_mcrl_reward_distribution

"""
Currently only tests costs to make sure they are being experienced
python3 -m unittest tests.test_generic_mouselab
"""

click_cost_tests_parameters = [
    # experiment setting, number trials, static cost weight, depth cost weight, 3-1-2 costs
    [
        "high_increasing",
        30,
        -79,
        80,
        [-1, -81, -161, -161, -1, -81, -161, -161, -1, -81, -161, -161],
    ],
    ["high_increasing", 30, 1, 0, [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
    ["high_increasing", 30, -1, 1, [0, -1, -2, -2, 0, -1, -2, -2, 0, -1, -2, -2]],
    ["high_increasing", 30, 0, 2, [-2, -4, -6, -6, -2, -4, -6, -6, -2, -4, -6, -6]],
]


class TestGenericMouselab(unittest.TestCase):
    @parameterized.expand(click_cost_tests_parameters)
    def test_click_costs(
        self, exp_setting, num_trials, static_cost, depth, resulting_costs
    ):
        branching = registry(exp_setting).branching
        reward_distributions = create_mcrl_reward_distribution(exp_setting)
        pipeline = construct_repeated_pipeline(
            branching, reward_distributions, num_trials
        )
        env = GenericMouselabEnv(
            num_trials,
            pipeline=pipeline,
            cost=[static_cost, depth],
        )
        costs = []
        for action in range(1, env.num_actions):
            _, cost, _, _ = env.step(action)
            costs.append(cost)

        self.assertTrue(costs == resulting_costs)
