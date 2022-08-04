from mcl_toolbox.utils.model_utils import ModelFitter
import numpy as np
import unittest


class TestModels(unittest.TestCase):
    model_index = [model_index for model_index in np.random.choice(range(10463), 50)]
    model_parameters = {
        "optimizer": "hyperopt",
        "num_simulations": 1,
        "max_evals": 1,
    }
    mf = ModelFitter("v1.0", number_of_trials=35)

    def test_learn_from_path_attribute(self):
        """
        Test the learn_from_path model attribute (see rlmodels.csv)
        """
        criterion = "pseudo_likelihood" #do not use likelihood as it won't work with the RSSL, SDSS models

        for model_index in self.model_index:
            print("model index", model_index)
            optimizer = self.mf.construct_optimizer(model_index, 1, criterion)
            _, _, _ = optimizer.optimize(criterion, **self.model_parameters)

            # what the agent gets boolean vs should be boolean according to rlmodels.csv
            self.assertTrue(optimizer.agent.learn_from_path_boolean == optimizer.learner_attributes["learn_from_path"])
