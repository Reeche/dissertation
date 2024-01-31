import os
from pathlib import Path

from mcl_toolbox.env.mouselab import MouselabEnv
from mcl_toolbox.env.conditional_mouselab import ConditionalMouselabEnv
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.global_vars import features, model, strategies, structure
from mcl_toolbox.mcrl_modelling.optimizer import ParameterOptimizer
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import (
    get_normalized_features,
    get_number_of_actions_from_branching,
    pickle_load,
    pickle_save,
    construct_repeated_pipeline,
    construct_reward_function,
    create_mcrl_reward_distribution
)
from mcl_toolbox.utils.sequence_utils import compute_log_likelihood

"""
Reinforce/LVOC: MF + habitual features
Habitual: MF + MB + habitual features
SSL: MF + MB + habitual features
Non-learning: MF + MB (without trial std that carries information from previous trials)
"""

all_features = features.implemented
non_habitual_features = features.microscope #contains both model free and model based features
non_learning_features = features.non_learning # no habitual but has model free and model based features
model_free_habitual_features = features.model_free_habitual # model free and habitual features
model_attributes = model.model_attributes
strategy_spaces = strategies.strategy_spaces


def get_strategy_probs(env, participant, features, normalized_features, w):
    strategy_probs = {}
    for trial_num in range(env.num_trials):
        chosen_strategy = participant.strategies[trial_num]
        strategy_weights = w[chosen_strategy - 1] * (1 / participant.temperature)
        ll = compute_log_likelihood(
            env.present_trial,
            participant.clicks[trial_num],
            features,
            strategy_weights,
            inv_t=False,
            normalized_features=normalized_features,
        )
        strategy_probs[trial_num] = ll
        env.get_next_trial()
    return strategy_probs


class ModelFitter:
    def __init__(self, exp_name, exp_attributes=None, data_path=None, **pipeline_kwargs):
        """
        
        :param exp_name: name, or folder, where experiment data is saved
        :param exp_attributes: dictionary, may contain:
            "block"          : None,  # Block of the experiment
            "exclude_trials" : None  # Trials to include
            and EITHER
            "experiment"     : Experiment object can be passed directly with pipeline and normalized features attached
            OR
            the experiment must be in the global_vars in the structure object
        :param data_path: path where data for experiment exp_name is saved
        """
        self.exp_name = exp_name
        if exp_attributes is None:
            exp_attributes = {
                "exclude_trials": None,
                "block": None,
                "experiment": None,
                "click_cost": 1,
                "learn_from_path": True,
            }
        if 'experiment' not in exp_attributes:
            exp_attributes['experiment'] = None

        if exp_attributes['experiment'] is not None:
            self.E = exp_attributes["experiment"]
            if self.E.pipeline is None:
                raise ValueError("Please attach pipeline to the experiment")
            if self.E.normalized_features is None:
                raise ValueError("Please attach normalized features to the experiment")
            self.pipeline = self.E.pipeline
            self.normalized_features = self.E.normalized_features
        else:  # if exp_attributes['experiment'] is None
            del exp_attributes['experiment']
            self.E = Experiment(self.exp_name, data_path=data_path, **exp_attributes)

            # Check if experiment, already in global_vars for backwards compatibility
            # check if pipeline.pkl already exist, mainly applicable for v1.0, c1.1, c1.0 and T1.1
            # if self.exp_name in structure.exp_pipelines.keys():
            #     self.pipeline = structure.exp_pipelines[self.exp_name]
            #     self.normalized_features = get_normalized_features(
            #         structure.exp_reward_structures[self.exp_name]
            #     )

            # if you want to add your experiment setting to global_vars.py instead of using the registry
            if self.exp_name in structure.branchings.keys():
                reward_dist = "categorical"
                reward_structure = structure.exp_reward_structures[self.exp_name]
                self.reward_distributions = construct_reward_function(
                    structure.reward_levels[reward_structure], reward_dist
                )
                repeated_pipeline = construct_repeated_pipeline(
                    structure.branchings[self.exp_name], self.reward_distributions, pipeline_kwargs["number_of_trials"]
                )
                self.pipeline = repeated_pipeline

            elif ("exp_setting" not in pipeline_kwargs) or ("num_trials" not in pipeline_kwargs):
                raise ValueError("Not enough information inputted to attach pipeline -- need exp_setting and "
                                 "num_trials")
            # if you want to use the registry to store experiment information
            else:
                self.reward_distributions = create_mcrl_reward_distribution(pipeline_kwargs["exp_setting"])
                branching = registry(pipeline_kwargs["exp_setting"]).branching
                self.pipeline = construct_repeated_pipeline(branching, self.reward_distributions,
                                                            pipeline_kwargs["number_of_trials"])
                self.normalized_features = get_normalized_features(pipeline_kwargs["exp_setting"])
            self.E.attach_pipeline(self.pipeline)
            self.normalized_features = get_normalized_features(
                structure.exp_reward_structures[self.exp_name]
            )
        self.branching = self.pipeline[0][0]
        self.num_actions = get_number_of_actions_from_branching(
            self.branching
        )  # Find out number of actions
        # excluded_trials = structure.excluded_trials[self.exp_name]
        self.participant = None
        self.env = None
        self.model_index = None
        if 'click_cost' in exp_attributes and exp_attributes['click_cost'] is not None:
            self.click_cost = exp_attributes['click_cost']
        else:
            self.click_cost = 1

    def update_attributes(self, env):
        self.pipeline = env.pipeline
        self.branching = self.pipeline[0][0]
        self.num_actions = get_number_of_actions_from_branching(self.branching)
        if env.normalized_features is not None:
            self.normalized_features = env.normalized_features

    def construct_env(self, participant, q_fn=None):
        if self.exp_name == "strategy_discovery":
            env = ConditionalMouselabEnv(
                num_trials=len(participant.envs),
                pipeline=self.pipeline,
                ground_truth=participant.envs,
                cost=self.click_cost,
                feedback=participant.condition,
                q_fn=q_fn,
            )
        else:
            env = GenericMouselabEnv(
                len(participant.envs),
                pipeline=self.pipeline,
                ground_truth=participant.envs,
                cost=self.click_cost,
                feedback=participant.condition,
                q_fn=q_fn,
            )
        return env

    def get_q_fn(self, participant):
        q_fn = None
        file_location = Path(__file__).parents[1]
        if hasattr(participant, "condition"):
            if participant.condition == "meta":
                try:
                    q_path = file_location.joinpath(f"data/{self.exp_name}_q.pkl")
                    q_fn = pickle_load(q_path)["q_dictionary"]
                except FileNotFoundError:
                    print("Q-fn could not be loaded")
        else:
            participant.condition = "none"
        return q_fn, participant

    def get_participant_context(self, pid):
        participant = self.E.participants[pid]
        q_fn, participant = self.get_q_fn(participant)
        env = self.construct_env(participant, q_fn=q_fn)
        return participant, env


    def construct_model(self, model_index):
        """
        1. Get model attributes from the rl_models.csv
        2. Attach selected features (if habitual, then full set of features (implemented_features.pkl; if not habitual,
        then microscope_features because the CM only uses the subset of features that are independent of what the
        participant did on the previous trials, i.e. no 5 habitual features)
        If it is the non-learning model, the non_learning_features.pkl is used, which contains one feature less than
        the microscope_features.pkl, namely trial_level_std

        Args:
            model_index: integer

        Returns:
            learner: str, e.g. "reinforce"
            learner_attributes: dict containing model features (list), normalized_features (list), num_priors (int),
            strategy_space (list of len 79), attributes that are from the rl_models.csv list

        """
        # get model attributes from global_vars.py, which gets it from models/rl_models.csv
        learner_attributes = model_attributes.iloc[model_index].to_dict()
        learner = learner_attributes["model"]
        strategy_space_type = learner_attributes["strategy_space_type"]
        strategy_space_type = (
            strategy_space_type if strategy_space_type else "microscope"
        )
        strategy_space = strategy_spaces[strategy_space_type]
        if learner_attributes["is_null"]:
            feature_space = non_learning_features
        else:
            if learner_attributes["habitual_features"] == "habitual":
                feature_space = model_free_habitual_features
                # feature_space = implemented_features
            else: # for the model not using habitual features
                feature_space = non_habitual_features
        if learner == "rssl":
            num_priors = 2 * len(strategy_space)
        else:
            num_priors = len(feature_space)
        pr_weight = learner_attributes["pr_weight"]
        if not pr_weight:  # todo: why?
            learner_attributes["pr_weight"] = 1
        learner_attributes = dict(
            features=feature_space,
            normalized_features=self.normalized_features,
            num_priors=num_priors,
            strategy_space=strategy_space,
            no_term=not learner_attributes["term"],
            num_actions=self.num_actions,
            **learner_attributes,
        )
        del learner_attributes["term"]  # todo: why is term deleted?
        return learner, learner_attributes

    def construct_optimizer(self, model_index, pid, optimization_criterion):
        # load experiment specific info
        learner, learner_attributes = self.construct_model(model_index)
        self.participant, self.env = self.get_participant_context(pid)
        # For likelihood fitting in case of RSSL models
        if optimization_criterion == "likelihood" and learner == "rssl":
            strategy_weights = strategies.strategy_weights
            strategy_probs = get_strategy_probs(
                self.env,
                self.participant,
                learner_attributes["features"],
                self.normalized_features,
                strategy_weights,
            )
            learner_attributes["strategy_probs"] = strategy_probs
        optimizer = ParameterOptimizer(
            learner, learner_attributes, self.participant, self.env
        )
        return optimizer

    def fit_model(
            self,
            model_index,
            pid,
            optimization_criterion,
            optimization_params,
            params_dir=None,
    ):
        self.model_index = model_index
        optimizer = self.construct_optimizer(model_index, pid, optimization_criterion)
        res, prior, obj_fn = optimizer.optimize(
            optimization_criterion, **optimization_params
        )
        losses = [trial["result"]["loss"] for trial in res[1]]
        print(f"Loss: {min(losses)}")
        if params_dir is not None:
            # save priors
            pickle_save(
                (res, prior),
                os.path.join(
                    params_dir, f"{pid}_{optimization_criterion}_{model_index}.pkl"
                ),
            )
        return res, prior, obj_fn

    def simulate_params(
            self,
            model_index,
            params,
            sim_params=None,
            env=None,
            pid=None,
            sim_dir=None,
            plot_dir=None,
    ):
        if sim_params is None:
            sim_params = {"num_simulations": 30,
                          "click_cost": 1}
        if env is None and pid is None:
            raise ValueError("Either env or pid has to be specified")
        num_simulations = sim_params["num_simulations"]
        click_cost = sim_params["click_cost"]
        participant = None
        if pid is not None:
            participant = self.E.participants[pid]
            q_fn, participant = self.get_q_fn(participant)
            env = self.construct_env(participant, q_fn=q_fn)
        self.update_attributes(env)
        learner, learner_attributes = self.construct_model(model_index)
        optimizer = ParameterOptimizer(
            learner, learner_attributes, participant=participant, env=env
        )
        if participant is None:
            (r_data, sim_data), p_data = optimizer.run_hp_model_nop(
                params, "reward", num_simulations=num_simulations
            )
            plot_file = f"{model_index}_{num_simulations}.png"
        else:
            (r_data, sim_data), p_data = optimizer.run_hp_model(
                params, "reward", num_simulations=num_simulations, click_cost=click_cost
            )
            plot_file = f"{participant.pid}_{model_index}_{num_simulations}.png"
        if plot_dir is not None:
            optimizer.reward_data = [r_data["mer"]]
            optimizer.p_data = p_data
            optimizer.plot_rewards(i=0, path=plot_dir.joinpath(plot_file))

        if sim_dir is not None:
            save_path = f"{participant.pid}_{model_index}_{num_simulations}.pkl"
            if participant is None:
                save_path = f"{model_index}_{num_simulations}.pkl"
            pickle_save(
                sim_data,
                sim_dir.joinpath(save_path),
            )
        return r_data, sim_data
