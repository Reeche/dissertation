from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.models.model_based_models import ModelBased
import unittest
# import pytest
import subprocess
import random
from mcl_toolbox.utils.analysis_utils import get_all_pid_for_env

def test_learning_pid():
    exp_name = "v1.0"
    E = Experiment("v1.0")
    pid = 1
    criterion = "likelihood"
    num_simulations = 1

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

    model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, test_fitted_model=False)

    model.test_fitted_model = True
    res = model.run_multiple_simulations({'inverse_temp': 0.5, 'sigma': 0.5})

    print("sampled action", model.sample_action())

    most_likely_node_value = {}
    for node, values in model.dirichlet_alpha_dict.items():
        most_likely_node_value[node] = max(model.dirichlet_alpha_dict[node], key=model.dirichlet_alpha_dict[node].get)

    # todo: make this dynamically pull from global_vars
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




def test_non_clicking_pid():
    exp_name = "v1.0"
    E = Experiment("v1.0")
    pid = 51
    criterion = "likelihood"
    num_simulations = 1

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

    model = ModelBased(env, value_range, participant_obj, criterion, num_simulations, test_fitted_model=False)

    model.test_fitted_model = True
    res = model.run_multiple_simulations({'inverse_temp': 0.5, 'sigma': 0.5})
    # because nothing clicked
    assert (p == 1 for p in model.dirichlet_alpha_dict.values())

class TestScript(unittest.TestCase):
    def test_multiple_runs_with_random_parameters(self):
        num_runs = 10
        for i in range(num_runs):
            # Sample random parameters
            exp_name = random.choice(['v1.0', 'c2.1', 'c1.1', 'high_variance_high_cost', 'high_variance_low_cost', 'low_variance_high_cost', 'low_variance_low_cost', 'strategy_discovery'])
            pid = random.sample(get_all_pid_for_env(exp_name), 1)[0]
            node_assumption = random.choice(['uniform', 'level', 'no_assumption'])
            update_rule = random.choice(['individual', 'level'])

            # exp_name = "high_variance_low_cost"
            # pid = "4"
            # node_assumption = "uniform"
            # update_rule = "individual"

            print(f"Testing with parameters: {exp_name}, {pid}, {node_assumption}, {update_rule}")

            cmd = ['python', '../fit_model_based_models.py', exp_name, 'likelihood', str(pid), node_assumption,
                   update_rule]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            self.assertEqual(process.returncode, 0, msg=f"Script returned an error: {stderr}")


if __name__ == '__main__':
    unittest.main()