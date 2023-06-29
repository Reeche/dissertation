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
    model.init_model_params()
    model.init_distributions()
    # model.env.reset()
    i = 0
    for _ in range(0,1):
        i += 1
        _ = model.simulate({'inverse_temp': 1})
        # print(i)
    return model.node_distributions,  model.dirichlet_alpha_dict

class TestModels(unittest.TestCase):
    distributions, alpha = test_model_based()
    most_likely_node_value = {}
    for node, values in alpha.items():
        most_likely_node_value[node] = max(alpha[node], key=alpha[node].get)

    #todo: make this dynamically pull from global_vars
    inner_nodes = [-4, -2, 2, 4]
    middle_nodes = [-8, -4, 4, 8]
    outer_nodes = [-48, -24, 24, 48]

    assert most_likely_node_value[1] in inner_nodes
    assert most_likely_node_value[2] in middle_nodes
    assert most_likely_node_value[3] in outer_nodes
    assert most_likely_node_value[4] in outer_nodes
    assert most_likely_node_value[5] in inner_nodes
    assert most_likely_node_value[6] in middle_nodes
    assert most_likely_node_value[7] in outer_nodes
    assert most_likely_node_value[8] in outer_nodes
    assert most_likely_node_value[9] in inner_nodes
    assert most_likely_node_value[10] in middle_nodes
    assert most_likely_node_value[11] in outer_nodes
    assert most_likely_node_value[12] in outer_nodes

