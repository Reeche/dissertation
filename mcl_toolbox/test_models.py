from mcl_toolbox.utils.model_utils import ModelFitter


def test_models(
        exp_name, pid, model_list, criterion="reward", optimization_params=None
):
    if optimization_params is None:
        optimization_params = {
            "optimizer": "hyperopt",
            "num_simulations": 1,
            "max_evals": 1,
        }
    mf = ModelFitter(exp_name)
    for model_index in model_list:
        print(model_index)
        optimizer = mf.construct_optimizer(model_index, pid, criterion)
        _, _, _ = optimizer.optimize(criterion, **optimization_params)
    return None


if __name__ == "__main__":
    # Testing if all the models at least run successfully
    test_models("v1.0", 0, range(6432))
