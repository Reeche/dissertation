from collections import defaultdict

import numpy as np
import scipy as sp

from mcl_toolbox.global_vars import hierarchical_params
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.utils.learning_utils import (get_log_norm_cdf,
                                              get_log_norm_pdf, rows_mean,
                                              temp_sigmoid)

precision_epsilon = hierarchical_params.precision_epsilon


class HierarchicalAgent:
    """Agent that performs the decision to terminate or continue"""

    def __init__(self, params, attributes):
        self.tau = np.exp(params["tau"])
        self.payoffs = []
        self.params = params
        self.decision_rule = attributes["decision_rule"]
        self.features = attributes["features"]
        self.max_payoff = 0
        self.avg_payoff = 0
        self.history = []
        self.action_log_probs = []
        self.compute_likelihood = False

    def update_payoffs(self, total_reward):
        self.payoffs.append(total_reward)
        avg_payoff = rows_mean(self.payoffs)
        self.avg_payoff = avg_payoff
        if total_reward > self.max_payoff:
            self.max_payoff = total_reward

    def update_history(self, max_expected_return):
        self.history.append(max_expected_return)

    def init_model_params(self):
        self.payoffs = []
        self.max_payoff = 0
        self.avg_payoff = 0
        self.history = []

    def compute_stop_prob(
        self,
        env,
        max_expected_return=0,
        max_payoff=0,
        avg_payoff=0,
        vpi=0,
        voi=0,
        max_improvement=0,
        expected_improvement=0,
        path_history=[0],
    ):
        decision_rule = self.decision_rule
        decision_params = self.params
        tau = self.tau
        if len(env.get_available_actions()) == 1:
            return 1.0
        if decision_rule == "threshold":
            max_return = env.present_trial.get_max_dist_value()
            min_return = env.present_trial.get_min_dist_value()
            max_min_diff = max_return - min_return
            normalized_return = (max_expected_return - min_return) / max_min_diff
            p_stop = temp_sigmoid(normalized_return - decision_params["theta"], tau)
        elif decision_rule == "best_payoff":
            p_stop = temp_sigmoid(
                max_expected_return - np.exp(decision_params["theta"]) * max_payoff, tau
            )
        elif decision_rule == "average_payoff":
            p_stop = temp_sigmoid(
                max_expected_return - np.exp(decision_params["theta"]) * avg_payoff, tau
            )
        elif decision_rule == "adaptive_satisficing":
            num_clicks = env.num_actions - len(env.get_available_actions())
            p_stop = temp_sigmoid(
                max_expected_return
                - np.exp(decision_params["a"])
                + np.exp(decision_params["b"]) * num_clicks,
                tau,
            )
        elif decision_rule == "feature":
            # Add adaptive satisficing
            normalized_feature_values = env.present_trial.node_map[
                0
            ].compute_termination_feature_values(self.features, adaptive_satisficing={})
            termination_feature_values = [
                normalized_feature_values[i] for i in self.termination_features
            ]
            decision_weights = [
                self.decision_params[f"f_{i}"] for i in range(len(self.decision_params))
            ]
            dot_product = np.dot(termination_feature_values, decision_weights)
            p_stop = temp_sigmoid(dot_product, tau)
        elif decision_rule == "best_path_difference":
            p_stop = temp_sigmoid(
                max_payoff - max_expected_return - decision_params["theta"], tau
            )
        elif decision_rule == "VPI":
            p_stop = temp_sigmoid(
                vpi - max_expected_return - decision_params["theta"], tau
            )
        elif decision_rule == "VOI1":
            p_stop = temp_sigmoid(
                voi - max_expected_return - decision_params["theta"], tau
            )
        elif decision_rule == "maximum_improvement":
            p_stop = temp_sigmoid(max_improvement - decision_params["theta"], tau)
        elif decision_rule == "expected_improvement":
            p_stop = temp_sigmoid(expected_improvement - decision_params["theta"], tau)
        elif decision_rule == "quantile":
            if decision_params["theta"] > 1:
                decision_params["theta"] = 1
            if decision_params["theta"] < 0:
                decision_params["theta"] = 0
            p_stop = temp_sigmoid(
                np.quantile(path_history, decision_params["theta"]), tau
            )
        elif decision_rule == "noisy_memory_best_payoff":
            alpha = decision_params["alpha"]
            beta = decision_params["beta"]
            p_stop = 0
            trial_num = env.present_trial_num
            for reward in set(path_history):
                p_forget_higher = 1
                p_forget_reward = 1
                for i, higher_reward in enumerate(path_history):
                    delta = trial_num - i
                    if higher_reward > reward:
                        p_forget_higher *= sp.stats.gamma.pdf(
                            delta, a=alpha, scale=beta
                        )
                    if higher_reward == reward:
                        p_forget_reward *= sp.stats.gamma.pdf(
                            delta, a=alpha, scale=beta
                        )
                p_remember_reward = 1 - p_forget_reward
                p_stop_conditional = temp_sigmoid(
                    max_expected_return - np.exp(decision_params["theta"]) * reward, tau
                )
                p_stop += p_remember_reward * p_forget_higher * p_stop_conditional
        elif decision_rule == "confidence_bound":
            trial_num = env.present_trial_num
            if trial_num == 0:
                threshold_mean = decision_params["threshold_mean"]
            else:
                threshold_mean = self.avg_payoff
            p_stop = temp_sigmoid(
                max_expected_return
                - np.random.normal(
                    threshold_mean,
                    decision_params["threshold_var"] / np.sqrt(trial_num + 1),
                ),
                tau,
            )
        return p_stop

    def get_action(self, env, trial_info):
        max_expected_return = env.get_term_reward()
        self.update_history(max_expected_return)
        p_stop = self.compute_stop_prob(
            env, max_expected_return=max_expected_return, path_history=self.history
        )
        probs = [p_stop, 1 - p_stop]
        if self.compute_likelihood:
            pi = trial_info["participant"]
            click = pi.get_click()
            termination_choice = 0
            if click != 0:
                termination_choice = 1
            prob = probs[termination_choice]
            if prob == 0:
                prob = precision_epsilon
            self.action_log_probs.append(np.log(prob))
        else:
            termination_choice = np.random.choice([0, 1], p=[p_stop, 1 - p_stop])
        return termination_choice, p_stop


class HierarchicalLearner(Learner):
    """ Two stage model of decision making"""

    def __init__(self, params, attributes):
        self.decision_rule = attributes["decision_rule"]
        self.actor = attributes["actor"]
        self.params = params
        self.init_weights = np.array(params["priors"])
        self.features = attributes["features"]
        self.num_features = len(self.features)
        self.normalized_features = attributes["normalized_features"]
        self.no_term = attributes["no_term"]
        self.decision_agent = HierarchicalAgent(params, attributes)
        self.actor_agent = self.actor(params, attributes)

    def simulate(self, env, compute_likelihood=False, participant=None):
        if compute_likelihood:
            get_log_norm_pdf.cache_clear()
            get_log_norm_cdf.cache_clear()
            self.actor_agent.compute_likelihood = True
            self.decision_agent.compute_likelihood = True
        self.compute_likelihood = compute_likelihood
        trials_data = defaultdict(list)
        num_trials = env.num_trials
        self.actor_agent.init_model_params()
        self.decision_agent.init_model_params()
        env.reset()
        self.actor_agent.num_actions = len(env.get_available_actions())
        for trial_num in range(num_trials):
            self.actor_agent.update_rewards = []
            self.actor_agent.update_features = []
            self.actor_agent.term_rewards = []
            self.actor_agent.previous_best_paths = []
            self.actor_agent.num_actions = len(env.get_available_actions())
            actions = []
            rewards = []
            trials_data["w"].append(self.actor_agent.get_current_weights())
            while True:
                continue_planning, _ = self.decision_agent.get_action(
                    env, trial_info={"participant": participant}
                )
                if continue_planning == 0:
                    previous_path = participant.get_trial_path()
                    _, reward, done, taken_path = env.step(0)
                    if compute_likelihood:
                        taken_path = previous_path
                    # What to do when episode is finished by upper level
                    _, act_reward, _, _ = self.actor_agent.act_and_learn(
                        env,
                        trial_info={
                            "end_episode": True,
                            "participant": participant,
                            "taken_path": taken_path,
                        },
                    )
                    actions.append(0)
                    if self.compute_likelihood:
                        reward = act_reward
                    rewards.append(reward)
                    actions.append(continue_planning)
                    break
                else:
                    action, reward, done, taken_path = self.actor_agent.act_and_learn(
                        env, trial_info={"participant": participant}
                    )
                    rewards.append(reward)
                    actions.append(action)
                    if done:
                        break
            env.get_next_trial()
            trials_data["a"].append(actions)
            trials_data["r"].append(np.sum(rewards))
        if self.decision_agent.action_log_probs and self.actor_agent.action_log_probs:
            trials_data["loss"] = -(
                np.sum(self.decision_agent.action_log_probs)
                + np.sum(self.actor_agent.action_log_probs)
            )
        else:
            trials_data["loss"] = None
        return dict(trials_data)
