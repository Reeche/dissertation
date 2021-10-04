import os

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.global_vars import *
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.sequence_utils import compute_log_likelihood

from mcl_toolbox.mcrl_modelling.optimizer import ParameterOptimizer
from mcl_toolbox.utils.learning_utils import (
    get_normalized_features, get_number_of_actions_from_branching,
    pickle_save)

implemented_features = features.implemented
microscope_features = features.microscope
model_attributes = model.model_attributes
strategy_spaces = strategies.strategy_spaces


def get_strategy_probs(env, participant, features, normalized_features, w):
    strategy_probs = {}
    for trial_num in range(env.num_trials):
        chosen_strategy = participant.strategies[trial_num]
        strategy_weights = w[chosen_strategy - 1] * (
                1 / participant.temperature
        )
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


class ModelFitter():
    def __init__(self, exp_name, exp_attributes=None):
        self.exp_name = exp_name
        self.normalized_features = get_normalized_features(
            structure.exp_reward_structures[self.exp_name]
        )
        self.pipeline = structure.exp_pipelines[self.exp_name]
        if exp_attributes is None:
            exp_attributes = {}
        self.E = Experiment(self.exp_name, **exp_attributes)
        self.E.attach_pipeline(self.pipeline)
        self.branching = structure.branchings[self.exp_name]
        self.num_actions = get_number_of_actions_from_branching(
            self.branching
        )  # Find out number of actions
        # excluded_trials = structure.excluded_trials[self.exp_name]
        self.participant = None
        self.env = None
        self.model_index = None

    def update_attributes(self, env):
        self.pipeline = env.pipeline
        self.branching = self.pipeline[0][0]
        self.num_actions = get_number_of_actions_from_branching(
            self.branching
        )
        if env.normalized_features is not None:
            self.normalized_features = env.normalized_features

    def construct_env(self, participant, q_fn=None):
        env = GenericMouselabEnv(
            len(participant.envs),
            pipeline=self.pipeline,
            ground_truth=participant.envs,
            feedback=participant.condition,
            q_fn=q_fn,
        )
        return env

    def get_q_fn(self, participant):
        q_fn = None
        if hasattr(participant, "condition"):
            if participant.condition == "meta":
                try:
                    q_path = os.path.join(file_location, f"data/{self.exp_name}_q.pkl")
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
        learner_attributes = model_attributes.iloc[model_index].to_dict()
        learner = learner_attributes["model"]
        strategy_space_type = learner_attributes["strategy_space_type"]
        strategy_space_type = strategy_space_type if strategy_space_type else "microscope"
        strategy_space = strategy_spaces[strategy_space_type]
        if learner_attributes["habitual_features"] == "habitual":
            feature_space = implemented_features
        else:
            feature_space = microscope_features
        if learner == "rssl":
            num_priors = 2 * len(strategy_space)
        else:
            num_priors = len(feature_space)
        pr_weight = learner_attributes["pr_weight"]
        if not pr_weight:
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
        del learner_attributes["term"]
        return learner, learner_attributes

    def construct_optimizer(self, model_index, pid, optimization_criterion):
        # load experiment specific info
        learner, learner_attributes = self.construct_model(model_index)
        self.participant, self.env = self.get_participant_context(
            pid)
        # For likelihood fitting in case of RSSL models
        if optimization_criterion == "likelihood" and learner == "rssl":
            strategy_weights = strategies.strategy_weights
            strategy_probs = get_strategy_probs(self.env, self.participant, learner_attributes['features'],
                                                self.normalized_features, strategy_weights)
            learner_attributes['strategy_probs'] = strategy_probs
        optimizer = ParameterOptimizer(learner, learner_attributes, self.participant, self.env)
        return optimizer

    def fit_model(self, model_index, pid, optimization_criterion, optimization_params, params_dir=None):
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

    def simulate_params(self, model_index, params, sim_params=None, env=None, pid=None,
                        sim_dir=None, plot_dir=None):
        if sim_params is None:
            sim_params = {'num_simulations': 30}
        if env is None and pid is None:
            raise ValueError("Either env or pid has to be specified")
        num_simulations = sim_params['num_simulations']
        participant = None
        if pid is not None:
            participant = self.E.participants[pid]
            q_fn, participant = self.get_q_fn(participant)
            env = self.construct_env(participant)
        self.update_attributes(env)
        learner, learner_attributes = self.construct_model(model_index)
        optimizer = ParameterOptimizer(learner, learner_attributes, participant=participant, env=env)
        if participant is None:
            (r_data, sim_data), p_data = optimizer.run_hp_model_nop(
                params, "reward", num_simulations=num_simulations
            )
            plot_file = f"{model_index}_{num_simulations}.png"
        else:
            (r_data, sim_data), p_data = optimizer.run_hp_model(
                params, "reward", num_simulations=num_simulations
            )
            plot_file = f"{participant.pid}_{model_index}_{num_simulations}.png"
        if plot_dir is not None:
            optimizer.reward_data = [r_data["mer"]]
            optimizer.p_data = p_data
            optimizer.plot_rewards(i=0, path=os.path.join(plot_dir,
                                   plot_file))

        if sim_dir is not None:
            pickle_save(
                sim_data,
                os.path.join(
                    sim_dir, f"{model_index}_{num_simulations}.pkl"
                ),
            )
        return r_data, sim_data