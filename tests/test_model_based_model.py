from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.models.model_based_models import ModelBased
import unittest

def test_model_based():
    exp_name = "v1.0"
    E = Experiment("v1.0")
    pid = 1
    # pid = 1  # 35 is okay adaptive  # 51#1 pid 1 is very adaptive

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": 1,
        "learn_from_path": True,
    }

    number_of_trials = 100

    p = E.participants[pid]
    participant_obj = ParticipantIterator(p)


    mf = ModelFitter(
        exp_name=exp_name,
        exp_attributes=exp_attributes,
        data_path=None,
        number_of_trials=number_of_trials)

    pid_context, env = mf.get_participant_context(pid)

    value_range = list(range(-120, 120))

    model = ModelBased(pid_context, env, value_range, True, participant_obj)
    model.compute_likelihood = False
    _ = model.simulate({'inverse_temp': 1})
    return model.node_distributions

class TestModels(unittest.TestCase):
    distributions = test_model_based()
    node_means = {}
    for node, value in distributions.items():
        node_means[node] = max(distributions[node].mean)
    outer_nodes = [node_means[3], node_means[4], node_means[7], node_means[8], node_means[11], node_means[12]]
    inner_nodes = [node_means[1], node_means[5], node_means[9]]
    middle_nodes = [node_means[2], node_means[6], node_means[10]]
    assert all(x > max(middle_nodes) for x in outer_nodes), "Mean of outer nodes are not larger than middle nodes"
    assert all(x > max(inner_nodes) for x in middle_nodes), "Mean of middle nodes are not larger than outer nodes"