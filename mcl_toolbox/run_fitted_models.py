from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from mcl_toolbox.utils.learning_utils import create_dir, pickle_load
from mcl_toolbox.utils.model_utils import ModelFitter
from simulate import cost_function, create_env


def plot_score(res, model):
    """Plot average score over trials."""
    plt.plot(np.mean(res["r"], axis=0), color="r", label="Model")
    plt.legend()
    plt.show()
    plt.close()


def plot_num_clicks(data, model, num_trials):
    """Plot number of clicks per trial."""
    model_num_clicks = []

    for pid_data in data.values():
        trial_clicks = {trial: 0 for trial in range(num_trials)}
        for sim in pid_data["a"]:
            for trial_idx, clicks in enumerate(sim):
                trial_clicks[trial_idx] += len(clicks) - 1

        avg_clicks = [trial_clicks[t] / len(pid_data["a"]) for t in range(num_trials)]
        model_num_clicks.append(avg_clicks)

    plt.plot(np.mean(model_num_clicks, axis=0), color="r", label=model)
    plt.legend()
    plt.savefig(f"{model}_num_clicks.png")
    plt.close()


def strategy_discovery_adap_prop(sim_data, model):
    """Plot binary adaptive reward pattern (reward in {13, 14, 15})."""
    rewards = sim_data["r"][0]
    binary_rewards = [1 if r in {13, 14, 15} else 0 for r in rewards]
    plt.plot(binary_rewards, color="b", label="Adaptive")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Configuration
    exp_name = "strategy_discovery"
    model_type = "hybrid"  # Options: "mf" or "hybrid"
    model_index = 491 if model_type == "mf" else 3316

    num_trials = 120
    num_simulations = 1
    plotting = False

    if exp_name in {"high_variance_high_cost", "low_variance_high_cost"}:
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    exp_attributes = {
        "exclude_trials": None,
        "block": None,
        "experiment": None,
        "click_cost": click_cost
    }

    sim_params = {
        "num_simulations": num_simulations,
        "click_cost": click_cost
    }

    fit_criterion = "likelihood"

    participant_ids = [58, 86, 238]  # Consider loading dynamically or from config

    for pid in participant_ids:
        parent_dir = Path(__file__).parents[1]
        param_path = parent_dir / f"results_sd_variant/mcrl/{exp_name}_priors"
        model_output_dir = parent_dir / f"results/mcrl/{exp_name}_data"
        create_dir(model_output_dir)

        plot_dir = None
        if plotting:
            plot_dir = parent_dir / f"results/mcrl/{exp_name}_plots"
            create_dir(plot_dir)

        model_fitter = ModelFitter(
            exp_name,
            exp_attributes=exp_attributes,
            data_path=f"results/mcrl/{exp_name}",
            number_of_trials=num_trials
        )

        try:
            result_data, prior_data = pickle_load(param_path / f"{pid}_{fit_criterion}_{model_index}.pkl")
        except FileNotFoundError:
            print(f"Could not load parameters for PID {pid}")
            continue

        r_data, sim_data = model_fitter.simulate_params(
            model_index,
            result_data[0],
            env=create_env(exp_name, num_trials),
            pid=pid,
            sim_params=sim_params,
            sim_dir=model_output_dir,
            plot_dir=plot_dir
        )

        # Uncomment below to analyze
        # strategy_discovery_adap_prop(sim_data, model_index)
        # plot_score(sim_data, model_index)
        # plot_num_clicks({pid: sim_data}, model_index, num_trials)

        # Optional: save r_data or sim_data to disk for later review
        # pickle_save(...)
