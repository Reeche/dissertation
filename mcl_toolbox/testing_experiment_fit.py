from mcl_toolbox.fit_mcrl_models import fit_model

from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import get_normalized_features
from mcl_toolbox.global_vars import structure

if __name__ == "__main__":
    exp_name = "c2.1_dec"
    E = Experiment("c2.1", variance=2424, block='test')
    normalized_features = get_normalized_features(structure.exp_reward_structures["c2.1_dec"])
    E.pipeline = structure.exp_pipelines["c2.1_dec"]
    E.normalized_features = normalized_features
    print(E.pids)
    exp_attributes = {'block': 'test', 'experiment': E}
    model_index = 1825
    optimization_criterion = "pseudo_likelihood"
    pid = 4
    num_simulations = 30
    simulate = True
    plotting = True
    optimization_params = {
        "optimizer": "hyperopt",
        "num_simulations": 1,
        "max_evals": 30
    }
    sim_params = {'num_simulations': num_simulations}
    fit_model(
        exp_name=exp_name,
        pid=pid,
        model_index=model_index,
        optimization_criterion=optimization_criterion,
        simulate=simulate,
        plotting=plotting,
        optimization_params=optimization_params,
        exp_attributes=exp_attributes
    )
