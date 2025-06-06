import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials

from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mcl_toolbox.models.model_based_models import ModelBased
from mcl_toolbox.utils.model_utils import ModelFitter
from mcl_toolbox.utils.experiment_utils import Experiment


def plot_score(res, pid):
    """Plot and save the average score (rewards) over time."""
    plt.plot(np.mean(res["rewards"], axis=0), color="r", label="Model")
    plt.legend()
    plt.title("Score")
    plt.savefig(f"score_{pid}.png")
    plt.close()


def plot_mer(res, pid):
    """Plot and save the average MER over time."""
    plt.plot(np.mean(res["mer"], axis=0), color="r", label="Model")
    plt.legend()
    plt.title("MER")
    plt.savefig(f"mer_{pid}.png")
    plt.close()


def plot_clicks(res, participant):
    """Plot and compare number of clicks per trial between participant and model."""
    pid_clicks = [len(action) - 1 for action in participant["a"]]
    model_clicks = [len(action) - 1 for action in res["a"][0]]

    plt.plot(pid_clicks, color="r", label="Participant")
    plt.plot(model_clicks, color="b", label="Model")
    plt.legend()
    plt.title("Clicks")
    plt.savefig("clicks_comparison.png")
    plt.close()


def cost_function(depth):
    """Cost function for depth-based strategies in the strategy_discovery experiment."""
    return {0: 0, 1: -1, 2: -3, 3: -30}.get(depth, 0)


def get_value_and_term_ranges(exp_name):
    """Return value and term ranges based on experiment configuration."""
    ranges = {
        "v1.0": ([-48, -24, -8, -4, -2, 2, 4, 8, 24, 48], list(range(-61, 61))),
        "c2.1": ([-48, -24, -8, -4, -2, 2, 4, 8, 24, 48], list(range(-61, 61))),
        "c1.1": ([-10, -5, 5, 10], list(range(-31, 31))),
        "high_variance_high_cost": ([-1000, -100, -50, -20, 50, 100], list(range(-3001, 3001))),
        "high_variance_low_cost": ([-1000, -100, -50, -20, 50, 100], list(range(-3001, 3001))),
        "low_variance_high_cost": ([-6, -4, -2, 2, 4, 6], list(range(-19, 19))),
        "low_variance_low_cost": ([-6, -4, -2, 2, 4, 6], list(range(-19, 19))),
        "strategy_discovery": ([-50, -5, -1, 1, 50], list(range(-47, 47))),
    }
    if exp_name not in ranges:
        raise ValueError(f"Experiment name '{exp_name}' not recognized")
    return ranges[exp_name]


def define_search_space(criterion, node_assumption):
    """Define parameter space for hyperparameter optimization based on modeling assumptions."""
    if criterion != "likelihood":
        return {
            'inverse_temp': hp.uniform('inverse_temp', -100, 100),
            'sigma': hp.uniform('sigma', np.log(1e-3), np.log(1e3)),
            'dist_alpha': hp.uniform('dist_alpha', 0, 5),
            'dist_beta': hp.uniform('dist_beta', 0, 5),
        }

    min_val, max_val = 0.01, 10
    click_weight = hp.uniform('click_weight', 1, 100)

    if node_assumption == "uniform":
        return {
            'inverse_temp': hp.uniform('inverse_temp', min_val, max_val),
            'alpha': hp.uniform('alpha', min_val, max_val),
            'beta': hp.uniform('beta', min_val, max_val),
            'click_weight': click_weight,
        }
    elif node_assumption == "level":
        return {
            'inverse_temp': hp.uniform('inverse_temp', min_val, max_val),
            **{f'{param}_{lvl}': hp.uniform(f'{param}_{lvl}', min_val, max_val)
               for lvl in range(1, 4) for param in ('alpha', 'beta')},
            'click_weight': click_weight,
        }
    elif node_assumption == "no_assumption":
        return {
            'inverse_temp': hp.uniform('inverse_temp', min_val, max_val),
            'click_weight': click_weight,
        }

    raise ValueError(f"Node assumption '{node_assumption}' not recognized")


def main():
    # Parse input arguments
    exp_name, criterion, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    node_assumption, update_rule = sys.argv[4], sys.argv[5]

    click_cost = (
        cost_function if exp_name == "strategy_discovery" else
        5 if "high_cost" in exp_name else
        1
    )

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": click_cost
    }

    num_trials = 120 if exp_name == "strategy_discovery" else 35
    num_simulations = 30 if criterion != "likelihood" else 1

    mf = ModelFitter(exp_name, exp_attributes, f"{exp_name}_mb", num_trials)
    pid_context, env = mf.get_participant_context(pid)
    participant_obj = ParticipantIterator(pid_context, click_cost=click_cost)

    value_range, term_range = get_value_and_term_ranges(exp_name)

    model = ModelBased(
        env, value_range, term_range, participant_obj, criterion, num_simulations,
        node_assumption, update_rule, compute_likelihood=True
    )

    fspace = define_search_space(criterion, node_assumption)

    trials = Trials()
    best_params = fmin(
        fn=model.run_multiple_simulations,
        space=fspace,
        algo=tpe.suggest,
        max_evals=60000,
        show_progressbar=True,
        trials=trials
    )

    # Re-run simulation with best parameters found
    model.compute_likelihood = False
    model.env.reset()
    model.participant_obj.reset()
    res = model.run_multiple_simulations(best_params)
    res.update(best_params)

    # Save results
    out_dir = Path(f"{exp_name}_mb")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}_{criterion}_{node_assumption}_{update_rule}.pkl"
    with open(out_path, 'wb') as output:
        pickle.dump(res, output)

    # Plot and save results
    plot_score(res, pid)
    plot_mer(res, pid)
    plot_clicks(res, model.p_data)
    print(res)


if __name__ == "__main__":
    main()
