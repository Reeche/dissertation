from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from mcl_toolbox.utils.learning_utils import create_dir, pickle_load
from mcl_toolbox.utils.model_utils import ModelFitter
from simulate import cost_function, create_env


def plot_score(res, model):
    plt.plot(np.mean(res["r"], axis=0), color="r", label="Model")
    plt.legend()
    plt.show()
    # plt.savefig(f"score_{model}.png")
    plt.close()
    return None


def plot_num_clicks(data, model, num_trials):
    # get the number of clicks by looking at the length of each sublist in res["a"]
    # init dict with key = trial, value = 0
    model_num_clicks = []
    for pid, value in data.items():
        num_clicks = {trial: 0 for trial in range(num_trials)}
        for simulation in value["a"]:
            for trial, sublist in enumerate(simulation):
                num_clicks[trial] += len(sublist) - 1

        # for every value in num_clicks, divide by num_simulations
        num_simulations = len(value["a"])
        for trial in num_clicks:
            num_clicks[trial] /= num_simulations
        model_num_clicks.append(list(num_clicks.values()))

    plt.plot(np.mean(model_num_clicks, axis=0), color="r", label=model)
    plt.legend()
    # plt.show()
    plt.savefig(f"{model}_num_clicks.png")
    plt.close()
    return None


def strategy_discovery_adap_prop(sim_data, model):
    # count how often the score was 13, 14, 15 for one participant/one simulation
    reward = sim_data["r"][0]
    # replace 13, 14, 15 with 1 and 0 otherwise
    reward_replaced = [1 if x in {13, 14, 15} else 0 for x in reward]

    # plot
    plt.plot(reward_replaced, color="b", label="Adaptive")
    plt.legend()
    plt.show()
    # plt.savefig(f"proportion_{model}.png")
    plt.close()


if __name__ == "__main__":
    # todo: this cannot run without a participant, therefore limited to the number of trials of participant
    exp_name = "strategy_discovery"
    type = "mf"

    if type == "mf":
        model_index = 491
    elif type == "hybrid":
        model_index = 3326

    num_trials = 120
    num_simulations = 1000
    plotting = False

    if exp_name == "high_variance_high_cost" or exp_name == "low_variance_high_cost":
        click_cost = 5
    elif exp_name == "strategy_discovery":
        click_cost = cost_function
    else:
        click_cost = 1

    exp_attributes = {
        "exclude_trials": None,  # Trials to be excluded
        "block": None,  # Block of the experiment
        "experiment": None,  # Experiment object can be passed directly with
        "click_cost": click_cost
    }

    sim_params = {"num_simulations": num_simulations,
                  "click_cost": click_cost}
    fit_criterion = "likelihood"

    # for pid in pid_dict[exp_name]:
    # for pid in [28, 35, 48, 63, 75, 195, 320, 335]:
    # [13, 62, 78, 141]
    for pid in [13]:
        parent_directory = Path(__file__).parents[1]
        param_dir = parent_directory.joinpath(f"final_results/{type}/{exp_name}_priors")
        # and directory to save fit model info in
        model_info_directory = parent_directory.joinpath(f"results/mcrl/{exp_name}_data")
        create_dir(model_info_directory)

        # add directory for reward plots, if plotting
        plot_directory = None
        if plotting:
            # plot_directory = parent_directory.joinpath(f"results_sd_test2/mcrl/{exp_name}_plots")
            plot_directory = parent_directory.joinpath(f"results/mcrl/{exp_name}_plots")
            create_dir(plot_directory)

        mf = ModelFitter(
            exp_name,
            exp_attributes=exp_attributes,
            data_path=f"results/mcrl/{exp_name}",
            number_of_trials=num_trials,
        )

        all_data = {}

        try:
            (res, prior) = pickle_load(
                param_dir.joinpath(f"{pid}_{fit_criterion}_{model_index}.pkl")
            )
        except:
            print(f"Could not load {pid}_{fit_criterion}_{model_index}.pkl")
            continue

        r_data, sim_data = mf.simulate_params(
            model_index,
            res[0],
            env=create_env(exp_name, num_trials),
            pid=pid,
            sim_params=sim_params,
            sim_dir=model_info_directory,
            plot_dir=plot_directory,
        )

        all_data[pid] = sim_data

        # strategy_discovery_adap_prop(sim_data, model_index)
        # plot_num_clicks(all_data, model_index, num_trials)
        # plot_score(sim_data, model_index)
        # print(pid)
        # print(r_data)
        # print(sim_data)
