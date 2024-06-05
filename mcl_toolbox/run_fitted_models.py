from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from mcl_toolbox.utils.learning_utils import create_dir, pickle_load
from mcl_toolbox.utils.model_utils import ModelFitter
from simulate import cost_function, create_env

pid_dict = {
    'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77,
             80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
             148, 150, 154, 155, 158, 160, 165, 169, 173],
    'c2.1': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
             79, 84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138,
             142, 145, 149, 152, 156, 162, 164, 166, 170, 172],
    'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
             83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
             151, 153, 157, 159, 161, 163, 167, 168, 171],
    'high_variance_high_cost': [0, 1, 10, 18, 22, 25, 30, 32, 38, 41, 46, 47, 49, 57, 60, 63, 65, 70, 74, 76, 81, 83,
                                88, 89, 94, 103, 108, 109, 111, 114, 116, 118, 125, 129, 134, 139, 149, 150, 156, 161,
                                164, 167, 169, 173, 177, 182, 188, 191, 195, 198, 199, 204],
    'high_variance_low_cost': [4, 7, 8, 17, 20, 23, 26, 29, 33, 35, 40, 48, 50, 51, 53, 56, 58, 64, 71, 78, 82, 87, 92,
                               93, 95, 96, 101, 112, 117, 119, 122, 126, 131, 133, 136, 141, 145, 146, 151, 154, 158,
                               162, 168, 175, 180, 185, 187, 189, 193, 197, 202, 205],
    'low_variance_high_cost': [2, 13, 14, 16, 21, 24, 28, 31, 36, 37, 43, 45, 54, 61, 62, 68, 69, 73, 79, 80, 84, 86,
                               90, 97, 98, 100, 102, 107, 120, 124, 128, 132, 135, 138, 140, 144, 147, 153, 157, 160,
                               163, 166, 171, 174, 181, 183, 192, 194, 201, 203, 206],
    'low_variance_low_cost': [3, 5, 6, 9, 11, 12, 15, 19, 27, 34, 39, 42, 44, 52, 55, 59, 66, 67, 72, 75, 77, 85, 91,
                              99, 104, 105, 106, 110, 113, 115, 121, 123, 127, 130, 137, 142, 143, 148, 152, 155, 159,
                              165, 170, 172, 176, 178, 179, 184, 186, 190, 196, 200, 207],
    'strategy_discovery': list(range(1, 379))}


def plot_score(res, model):
    plt.plot(np.mean(res["r"], axis=0), color="r", label="Model")
    plt.legend()
    # plt.show()
    plt.savefig(f"score_{model}.png")
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
    # plt.show()
    plt.savefig(f"proportion_{model}.png")
    plt.close()


if __name__ == "__main__":
    # todo: this cannot run without a participant, therefore limited to the number of trials of participant
    exp_name = "strategy_discovery"
    model_index = 32
    num_trials = 200
    num_simulations = 1
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
    for pid in [172]:
        parent_directory = Path(__file__).parents[1]
        param_dir = parent_directory.joinpath(f"results_sd_test2/mcrl/{exp_name}_priors")
        # and directory to save fit model info in
        model_info_directory = parent_directory.joinpath(f"results_sd_test2/mcrl/{exp_name}_data")
        create_dir(model_info_directory)

        # add directory for reward plots, if plotting
        plot_directory = None
        if plotting:
            plot_directory = parent_directory.joinpath(f"results_sd_test2/mcrl/{exp_name}_plots")
            create_dir(plot_directory)

        mf = ModelFitter(
            exp_name,
            exp_attributes=exp_attributes,
            data_path=f"results/mcrl/{exp_name}",
            number_of_trials=num_trials,
        )

        all_data = {}

        (res, prior) = pickle_load(
            param_dir.joinpath(f"{pid}_{fit_criterion}_{model_index}.pkl")
        )

        r_data, sim_data = mf.simulate_params(
            model_index,
            res[0],
            env=create_env(exp_name, num_trials),
            pid=None,
            sim_params=sim_params,
            sim_dir=model_info_directory,
            plot_dir=plot_directory,
        )

        all_data[pid] = sim_data

        # strategy_discovery_adap_prop(sim_data, model_index)
        plot_num_clicks(all_data, model_index, num_trials)
        plot_score(sim_data, model_index)
        # print(pid)
        # print(r_data)
        # print(sim_data)
