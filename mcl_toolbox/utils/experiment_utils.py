import itertools
import operator
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from statsmodels.stats.proportion import proportions_chisquare
from mcl_toolbox.utils.analysis_utils import get_data
from mcl_toolbox.utils.learning_utils import sidak_value
from mcl_toolbox.utils.sequence_utils import get_acls

# Matplotlib no grid
plt.rcParams["axes.grid"] = False

# For now, it is set to ignore
np.seterr(all='ignore')


class Participant():
    # TODO:
    # Add proper way of managing temperature pararmeters
    def __init__(self, pid, condition=None):
        self.pid = pid
        self.condition = condition
        self.strategies = None

    def modify_clicks(self):
        modified_clicks = []
        click_sequence = self.clicks
        for clicks in click_sequence:
            modified_clicks.append([int(c) for c in clicks] + [0])
        self.clicks = modified_clicks

    def attach_trial_data(self, data):
        self.clicks = [q['click']['state']['target'] for q in data.queries]
        self.modify_clicks()
        self.envs = [[0] + sr[1:] for sr in data.state_rewards]
        columns = list(data.columns).copy()
        columns_to_remove = ['pid', 'queries', 'state_rewards']
        # make it list of rewards
        for col in columns_to_remove:
            columns.remove(col)
        for attr in columns:
            setattr(self, attr, data[attr].values)

    def attach_strategies(self, strategies):
        self.strategies = strategies

    def attach_temperature(self, temperature):
        self.temperature = temperature

    def attach_feature_properties(self, features, normalized_features, strategy_weights):
        self.features = features
        self.normalized_features = normalized_features
        self.feature_weights = np.array([strategy_weights[s - 1] for s in self.strategies])

    def attach_decision_system_properties(self, decision_systems, decision_system_weights,
                                          decision_system_proportions):
        self.decision_systems = decision_systems
        self.decision_system_weights = np.array([decision_system_weights[s - 1] for s in self.strategies])
        self.decision_system_proportions = np.array([decision_system_proportions[s - 1] for s in self.strategies])

    def attach_clusters(self, cluster_map):
        self.clusters = [cluster_map[s] for s in self.strategies]


class Experiment():
    """
    This class contains all plots and analysis with regards to the Computational Micropscope
    """
    def __init__(self, exp_num, cm=None, pids=None, block=None, **kwargs):
        self.exp_num = exp_num
        self.data = get_data(exp_num)
        self.cm = cm
        self.block = None
        if pids:
            self.pids = pids
        else:
            if hasattr(self.data, 'pids'):
                self.pids = self.data['pids']
            else:
                self.pids = sorted(np.unique(self.data['participants']['pid']).tolist())
        self.participants = {}
        if block:
            self.block = block
        self.additional_constraints = kwargs
        self.init_participants()
        self.init_planning_data()
        self.participant_strategies = {}
        self.participant_temperatures = {}

    def init_participants(self):
        participants_data = self.data['participants']
        self.conditions = set()
        for constraint in self.additional_constraints.keys():
            participants_data = participants_data[
                participants_data[constraint] == self.additional_constraints[constraint]]
            pids = participants_data['pid'].tolist()
            self.pids = [p for p in pids if p in self.pids]
        trial_nums = []
        for pid in self.pids:
            p_data = participants_data[participants_data.pid == pid]
            if hasattr(p_data, 'condition'):
                condition = p_data.condition.values[0]
                self.conditions.add(condition)
            else:
                if hasattr(p_data, 'feedback'):
                    condition = p_data.feedback.values[0]
                    self.conditions.add(condition)
                else:
                    condition = None  # Verify this
            p_trials_data = self.data['mouselab-mdp']
            if not self.block:
                p_trials_data = p_trials_data[(p_trials_data.pid == pid)]
            else:
                p_trials_data = p_trials_data[(p_trials_data.pid == pid) & (p_trials_data.block == self.block)]

            p = Participant(pid, condition)
            trial_nums.append(len(p_trials_data))
            p.attach_trial_data(p_trials_data)
            p.condition = condition
            self.participants[pid] = p
        self.num_trials = max(trial_nums, key=trial_nums.count)

    def init_planning_data(self):
        self.planning_data = defaultdict(lambda: dict())
        for pid in self.pids:
            self.planning_data['envs'][pid] = self.participants[pid].envs
            self.planning_data['clicks'][pid] = self.participants[pid].clicks
        self.planning_data = dict(self.planning_data)

    def infer_strategies(self, precomputed_strategies=None, precomputed_temperatures=None, max_evals=30,
                         show_pids=True):
        leftout_pids = []
        cm = self.cm
        pids = []
        if precomputed_strategies:
            for pid in precomputed_strategies.keys():
                if show_pids:
                    print("SHOW PID", pid)
                try:
                    S = precomputed_strategies[pid]
                    self.participants[pid].attach_strategies(S)
                    self.participant_strategies[pid] = S
                    if precomputed_temperatures:
                        T = precomputed_temperatures[pid]
                        self.participants[pid].attach_temperature(T)
                        self.participant_temperatures[pid] = T
                    pids.append(pid)
                except KeyError:
                    print(f"Strategies for {pid} not found. Skipping adding strategy data")
        else:
            if not cm:
                raise ValueError("Computational Microscope not found.")
            else:
                self.participant_strategies, self.participant_temperatures = cm.infer_participant_sequences(self.pids,
                                                                                                            self.planning_data[
                                                                                                                'envs'],
                                                                                                            self.planning_data[
                                                                                                                'clicks'],
                                                                                                            max_evals=max_evals,
                                                                                                            show_pids=show_pids)
                for pid in self.participant_strategies:
                    self.participants[pid].attach_strategies(self.participant_strategies[pid])
                    self.participants[pid].attach_temperature(self.participant_temperatures[pid])
                    pids.append(pid)
        self.pids = pids

    def get_transition_frequencies(self, trial_wise=False, pids=None, clusters=False):
        """
        Counting the occurance of pairwise strategies
        Args:
            trial_wise:
            pids:
            clusters:

        Returns:

        """
        if not self.participant_strategies:
            raise (ValueError("Please update participant strategies"))
        if not pids:
            pids = self.pids
        if pids:
            if not clusters:
                sequences = np.array([self.participant_strategies[pid] for pid in pids])
            else:
                sequences = np.array([self.participant_clusters[pid] for pid in pids])
        if not trial_wise:
            strategy_pairs = []
            for S in sequences:
                for first, second in zip(S, S[1:]):
                    strategy_pairs.append((first, second))
            strategy_pair_counts = Counter(strategy_pairs)
        else:
            strategy_pairs = defaultdict(list)
            for S in sequences:
                for i, (first, second) in enumerate(zip(S, S[1:])):
                    strategy_pairs[i].append((first, second))
            strategy_pair_counts = {k: Counter(v) for k, v in strategy_pairs.items()}
        return strategy_pair_counts

    def perform_chi2_conditions(self, t_a, t_b):
        t_a = defaultdict(int, t_a)
        t_b = defaultdict(int, t_b)
        n_a = sum(list(t_a.values()))
        n_b = sum(list(t_b.values()))
        all_transitions = set([*(t_a.keys()), *(t_b.keys())])
        num_transitions = len(all_transitions)
        alpha_sidak = sidak_value(0.05, num_transitions)
        significant_transitions = []
        insignificant_transitions = []
        for transition in all_transitions:
            f_a = np.round(t_a[transition] / n_a, 4) if n_a else 0
            f_b = np.round(t_b[transition] / n_b, 4) if n_b else 0
            freq = (f_a, f_b)
            result = proportions_chisquare([t_a[transition], t_b[transition]], nobs=[n_a, n_b])
            chi2 = np.round(result[0], 2)
            p = np.round(result[1], 4)
            if p < alpha_sidak and p != np.nan:
                significant_transitions.append((transition, freq, chi2, p))
            else:
                insignificant_transitions.append((transition, freq, chi2, p))
        return significant_transitions, insignificant_transitions, alpha_sidak

    def strategy_transitions_chi2(self, trial_wise=False, clusters=False, print_results=True):
        condition_wise_pids = defaultdict(list)
        for pid in self.pids:
            condition_wise_pids[self.participants[pid].condition].append(pid)
        # condition_wise_transitions = {
        #     k: self.get_transition_frequencies(trial_wise=trial_wise, pids=v, clusters=clusters)
        #     for k, v in condition_wise_pids.items()}
        # copied line below from the original CM repo
        condition_wise_transitions = {k: self.get_transition_frequencies(trial_wise=trial_wise, pids=v, clusters=clusters) for k, v in condition_wise_pids.items()}
        conditions = list(condition_wise_pids.keys())
        condition_combinations = list(itertools.combinations(conditions, 2))
        results = defaultdict(lambda: defaultdict())
        for comb in condition_combinations:
            significant_transitions, insignificant_transitions, alpha_sidak = self.perform_chi2_conditions(
                condition_wise_transitions[comb[0]], condition_wise_transitions[comb[1]])
            results[comb]['significant'] = significant_transitions
            results[comb]['insignificant'] = insignificant_transitions
            results[comb]['alpha_sidak'] = alpha_sidak
            if print_results:
                print(comb)
                print("Number of significant transitions:", len(significant_transitions))
                print("Number of insignificant transitions:", len(insignificant_transitions))
                print("Alpha sidak:", alpha_sidak)
                print("Significant transitions:", significant_transitions)
                print("\n")
        return results

    # The strategy scores should vary with reward structure
    # Catch error when there is no significant transition
    def performance_transitions_chi2(self, strategy_scores=None, cluster_scores=None, trial_wise=False):
        performance_results = defaultdict(lambda: defaultdict())
        if strategy_scores:
            scores = strategy_scores
            chi2_results = self.strategy_transitions_chi2(trial_wise=trial_wise, clusters=False, print_results=False)
        if cluster_scores:
            scores = cluster_scores
            chi2_results = self.strategy_transitions_chi2(trial_wise=trial_wise, clusters=True, print_results=False)

        for condition_pair in chi2_results.keys():
            significant_transitions = chi2_results[condition_pair]['significant']
            performance_results[condition_pair]['increasing'] = [t for t in significant_transitions if
                                                                 scores[t[0][0]] < scores[t[0][1]]]
            performance_results[condition_pair]['decreasing'] = [t for t in significant_transitions if
                                                                 scores[t[0][0]] > scores[t[0][1]]]
            performance_results[condition_pair]['constant'] = [t for t in significant_transitions if
                                                               scores[t[0][0]] == scores[t[0][1]]]

        for comb in chi2_results.keys():
            print(comb)
            print("Performance Increasing transitions", performance_results[comb]['increasing'])
            print("Performance Decreasing transitions", performance_results[comb]['decreasing'])
            print("Constant transitions", performance_results[comb]['constant'])
            print("\n")
        return performance_results

    # Should we also add constant case?
    def frequency_transitions_chi2(self, clusters=False, trial_wise=False):
        frequency_results = defaultdict(lambda: defaultdict())
        chi2_results = self.strategy_transitions_chi2(trial_wise=trial_wise, clusters=clusters, print_results=False)
        for condition_pair in chi2_results.keys():
            significant_transitions = chi2_results[condition_pair]['significant']
            frequency_results[condition_pair]['increasing'] = [t for t in significant_transitions if t[1][0] < t[1][1]]
            frequency_results[condition_pair]['decreasing'] = [t for t in significant_transitions if t[1][0] > t[1][1]]

        for comb in chi2_results.keys():
            print(comb)
            print("Frequency Increasing transitions", frequency_results[comb]['increasing'])
            print("Frequency Decreasing transitions", frequency_results[comb]['decreasing'])
            print("\n")
        return frequency_results

    def make_trajectory(self, strategy_sequence):
        previous_strategy = -1
        trajectory = []
        for s in strategy_sequence:
            if s != previous_strategy:
                trajectory.append(s)
                previous_strategy = s
        return tuple(trajectory)

    def get_trajectories(self, clusters=False, pids=None):
        if not pids:
            pids = self.pids
        if not clusters:
            sequences = np.array([self.participant_strategies[pid] for pid in pids])
        else:
            sequences = np.array([self.participant_clusters[pid] for pid in pids])
        trajectories = [self.make_trajectory(S) for S in sequences]
        return trajectories

    def get_trajectory_counts(self, clusters=False, pids=None):
        trajectories = self.get_trajectories(clusters, pids)
        trajectory_counts = Counter(trajectories)
        # print(sorted(trajectory_counts.items(), key=operator.itemgetter(1), reverse=True))
        return trajectory_counts

    def get_condition_trajectory_counts(self, clusters=False):
        condition_wise_pids = defaultdict(list)
        condition_trajectory_counts = {}
        for pid in self.pids:
            condition_wise_pids[self.participants[pid].condition].append(pid)
        for condition in self.conditions:
            trajectory_counts = self.get_trajectory_counts(clusters=clusters, pids=condition_wise_pids[condition])
        condition_trajectory_counts[condition] = trajectory_counts
        return condition_trajectory_counts

    def get_paths_to_optimal(self, clusters=False, optimal_S=21, optimal_C=10):
        trajectory_counts = self.get_trajectory_counts(clusters=clusters)
        total_trajectories = sum(list(trajectory_counts.values()))
        optimal_trajectories = {}
        penultimate_strategies = []
        for t in trajectory_counts.keys():
            if not clusters:
                if t[-1] == optimal_S:
                    optimal_trajectories[t] = trajectory_counts[t]
                    if len(t) > 1:
                        penultimate_strategies += [t[-2]] * trajectory_counts[t]
            else:
                if t[-1] == optimal_C:
                    optimal_trajectories[t] = trajectory_counts[t]
                    if len(t) > 1:
                        penultimate_strategies += [t[-2]] * trajectory_counts[t]
        print(sorted(Counter(penultimate_strategies).items(), key=operator.itemgetter(1), reverse=True))
        print(sorted(optimal_trajectories.items(), key=operator.itemgetter(1), reverse=True))
        return optimal_trajectories

    def init_feature_properties(self, features, normalized_features, strategy_weights):
        if not hasattr(self, 'participant_strategies'):
            raise ValueError("Strategies not found. Please initialize strategies before initializing the weights.")
        no_inference = False
        self.features = features
        self.normalized_features = normalized_features
        self.strategy_weights = strategy_weights  # These contain weights of all the 89 strategies.
        for pid in self.pids:
            if not hasattr(self.participants[pid], 'strategies'):
                print(f"Strategies for participant {pid} not found.")
                no_inference = True
            else:
                self.participants[pid].attach_feature_properties(features, normalized_features, strategy_weights)
        if no_inference:
            self.infer_strategies(precomputed_strategies=self.participant_strategies)
            self.init_feature_properties(features, normalized_features, strategy_weights)

    def init_decision_system_properties(self, decision_systems, decision_weights, decision_proportions):
        if not hasattr(self, 'participant_strategies'):
            raise ValueError("Strategies not found. Please initialize strategies before initializing\
                                    the weights.")
        no_inference = False
        self.decision_systems = decision_systems
        for pid in self.pids:
            if not hasattr(self.participants[pid], 'strategies'):
                print(f"Strategies for participant {pid} not found.")
                # del self.participants[pid]
                # self.pids.remove(pid)
                # no_inference = True
            self.participants[pid].attach_decision_system_properties(decision_systems, decision_weights,
                                                                     decision_proportions)
        # if no_inference:
        #     self.infer_strategies(precomputed_strategies=self.participant_strategies)
        #     self.init_decision_system_properties(decision_systems, decision_weights, decision_proportions)

    def plot_average_ds(self, suffix=""):
        """
        Plot the averaged decision system proportions per trial
        Args:
            suffix:

        Returns: plots

        """
        DSP = []
        num_trials = self.num_trials
        for pid in self.pids:
            decision_systems = self.participants[pid].decision_systems
            ds_prop = self.participants[pid].decision_system_proportions
            if len(ds_prop) == num_trials:
                DSP.append(ds_prop)
        decision_system_labels = ["Mental effort avoidance", "Model-based Metareasoning",
                                  "Model-free values and heuristics",
                                  "Pavlovian", "Satisficing and stopping"]
        # decision_system_labels = [" ".join([s.capitalize() for s in d.split("_")]) for d in decision_systems]
        num_decision_systems = len(decision_systems)
        mean_dsw = np.mean(DSP, axis=0)
        fig = plt.figure(figsize=(16, 10))
        for i in range(num_decision_systems):
            plt.plot(range(1, num_trials + 1), mean_dsw[:, i], label=decision_system_labels[i], linewidth=3.0)
        plt.xlabel("Trial Number", size=24)
        plt.tick_params(labelsize=22)
        plt.ylabel("Relative Influence of Decision System", fontsize=24)
        # plt.title("Decision system proportions", fontsize=24)
        plt.ylim(top=np.max(mean_dsw) + 0.2)
        plt.legend(prop={'size': 22}, ncol=2, loc='upper center')
        plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_decision_plots_{suffix}.png",
                    bbox_inches='tight')
        plt.close(fig)
        # print(mean_dsw.shape)
        return mean_dsw

    def get_proportions(self, strategies, trial_wise=False):
        strategies_list = [strategies[pid] for pid in self.pids]
        total_S = []
        for S in strategies_list:
            total_S += S
        if not trial_wise:
            num_strategies = len(total_S)
            strategy_counts = Counter(total_S)
            strategy_proportions = {k: v / num_strategies for k, v in strategy_counts.items()}
        else:
            strategy_proportions = defaultdict(lambda: defaultdict(int))
            for S in strategies_list:
                for i, s in enumerate(S):
                    strategy_proportions[i][s] += 1
            for i in strategy_proportions.keys():
                strategy_proportions[i] = dict(strategy_proportions[i])
                total_v = sum(list(strategy_proportions[i].values()))
                strategy_proportions[i] = {k: v / total_v for k, v in strategy_proportions[i].items()}
            strategy_proportions = dict(strategy_proportions)
        return strategy_proportions

    def get_strategy_frequencies(self, strategies, trial_wise=False):
        strategies_list = [strategies[pid] for pid in self.pids]
        total_S = []
        for S in strategies_list:
            total_S += S
        if not trial_wise:
            num_strategies = len(total_S)
            strategy_counts = Counter(total_S)
            total = num_strategies
        else:
            strategy_counts = defaultdict(lambda: defaultdict(int))
            for S in strategies_list:
                for i, s in enumerate(S):
                    strategy_counts[i][s] += 1
            total = np.zeros(len(strategy_counts.keys()))
            for i in strategy_counts.keys():
                strategy_counts[i] = dict(strategy_counts[i])
                total_v = sum(list(strategy_counts[i].values()))
                total[i] = total_v
            strategy_counts = dict(strategy_counts)
        return strategy_counts, total

    def get_strategy_proportions(self, trial_wise=False):
        if not trial_wise:
            if hasattr(self, 'strategy_proportions'):
                return self.strategy_proportions
        else:
            if hasattr(self, 'trial_strategy_proportions'):
                return self.trial_strategy_proportions
        strategy_proportions = self.get_proportions(self.participant_strategies, trial_wise=trial_wise)
        if not trial_wise:
            self.strategy_proportions = strategy_proportions
        else:
            self.trial_strategy_proportions = strategy_proportions
        return strategy_proportions

    def get_adjusted_strategy_proportions(self, trial_wise=False, confusions={}):
        proportions = self.get_strategy_proportions(trial_wise=trial_wise)
        adjusted_proportions = defaultdict(int)
        if not trial_wise:
            for k, v in proportions.items():
                for s in confusions[k].keys():
                    adjusted_proportions[s] += v * confusions[k][s]
            adjusted_proportions = dict(adjusted_proportions)
        else:
            adjusted_proportions = {}
            for t_num in proportions.keys():
                t_prop = defaultdict(int)
                for k, v in proportions[t_num].items():
                    for s in confusions[k].keys():
                        t_prop[s] += v * confusions[k][s]
                adjusted_proportions[t_num] = dict(t_prop)
        if not trial_wise:
            self.adjusted_strategy_proportions = adjusted_proportions
        else:
            self.adjusted_trial_strategy_proportions = adjusted_proportions
        return adjusted_proportions

    def plot_proportions(self, trial_prop, S, title="", suffix="", labels=[], cluster=False,
                         combine_other=False):
        S_proportions = []
        # the lines below are from original CM repo
        if cluster and 13 in S:
            index = S.index(13)
        else:
            index = None
        # the lines below are from original CM repo
        for t in trial_prop.keys():
            props = []
            for s in S:
                if not index or (index and s != 13):
                    props.append(trial_prop[t].get(s, 0))
            if index:
                if combine_other:
                    total_prop = np.sum(props)
                    props.insert(index, 1-total_prop)
                else:
                    props.insert(index, trial_prop[t].get(13, 0))
            S_proportions.append(props)
        S_proportions = np.array(S_proportions)
        #labels = ["Myopic Forward Planning", "Goal setting with additional immediate exploration", "Postive satisificing with two additional nodes", "Exploring parent of best leaf", "No planning", "Optimal Planning"]

        fig = plt.figure(figsize=(16, 10))
        prefix = "Strategy"
        if cluster:
            prefix = "Cluster"
        # the lines below are from original CM repo
        for i in range(S_proportions.shape[1]):
            if labels:
                label = labels[i]
            else:
                label = f"{prefix} {S[i]}"
            plt.plot(range(1, S_proportions.shape[0]+1), S_proportions[:, i]*100, label=label, linewidth=3.0)
        plt.xlabel("Trial Number", fontsize=28)
        plt.ylabel("Proportion (%)", fontsize=28)
        #plt.title(title, fontsize=24)
        if not cluster:
            plt.ylim(top=95)
        else:
            plt.ylim(top=95)
        plt.tick_params(labelsize=22)
        plt.legend(prop={'size': 22}, ncol=3, loc='upper center')
        if cluster:
            plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_cluster_proportions_{suffix}.png",
                        dpi=400, bbox_inches='tight')
        else:
            plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_strategy_proportions_{suffix}.png",
                        dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    def plot_strategy_proportions_pertrial(self, S, suffix="", labels = None):
        if not hasattr(self, 'trial_strategy_proportions'):
            self.get_strategy_proportions(trial_wise=True)
        self.plot_proportions(self.trial_strategy_proportions, S, title="Strategy proportions", suffix=suffix, labels=labels)

    ### Emperical validations
    def plot_strategy_scores(self, strategy_scores):
        """
        I think this one only works for the increasing variance environment (see input)
        # todo: create strategy scores for all environment if you want to use this function
        Args:
            strategy_scores:  average score of each strategy on the increasing variance environment

        Returns:

        """
        # This is a sanity check
        if not hasattr(self, 'participant_strategy_scores'):
            self.participant_strategy_scores = {pid: [strategy_scores[s] for s in self.participants[pid].strategies] for
                                                pid
                                                in self.pids}
        num_trials = self.num_trials  # Change this
        scores = list(self.participant_strategy_scores.values())
        data = []
        for score in scores:
            for i, s in enumerate(score):
                data.append([i, s])
        df = pd.DataFrame(data, columns=["trial", r"$Score_t$"])
        sns.lmplot(x="trial", y=r"$Score_t$", data=df)
        plt.title("Score as a function of trial number")

    def init_strategy_clusters(self, cluster_map):
        self.participant_clusters = {}
        for pid in self.pids:
            self.participants[pid].attach_clusters(cluster_map)
            self.participant_clusters[pid] = self.participants[pid].clusters

    # Fix this
    def get_cluster_proportions(self, trial_wise=False):
        if not trial_wise:
            if hasattr(self, 'cluster_proportions'):
                return self.cluster_proportions
        else:
            if hasattr(self, 'trial_cluster_proportions'):
                return self.trial_cluster_proportions
        cluster_proportions = self.get_proportions(self.participant_clusters, trial_wise=trial_wise)
        if not trial_wise:
            self.cluster_proportions = cluster_proportions
        else:
            self.trial_cluster_proportions = cluster_proportions
        return cluster_proportions

    def get_adjusted_cluster_proportions(self, trial_wise=False, confusions={}):
        proportions = self.get_cluster_proportions(trial_wise=trial_wise)
        adjusted_proportions = defaultdict(int)
        if not trial_wise:
            for k, v in proportions.items():
                for s in confusions[k].keys():
                    adjusted_proportions[s] += v * confusions[k][s]
            adjusted_proportions = dict(adjusted_proportions)
        else:
            adjusted_proportions = {}
            for t_num in proportions.keys():
                t_prop = defaultdict(int)
                for k, v in proportions[t_num].items():
                    for s in confusions[k].keys():
                        t_prop[s] += v * confusions[k][s]
                adjusted_proportions[t_num] = dict(t_prop)

        if not trial_wise:
            self.adjusted_cluster_proportions = adjusted_proportions
        else:
            self.adjusted_trial_cluster_proportions = adjusted_proportions

        return adjusted_proportions

    def plot_cluster_proportions(self, C, suffix="", labels=None, combine_other=False):
        if not hasattr(self, 'trial_cluster_proportions'):
            cluster_proportions = self.get_cluster_proportions(trial_wise=True)
        self.plot_proportions(self.trial_cluster_proportions, C, title="Cluster Proportions",
                              suffix = suffix, labels=labels, cluster=True,
                              combine_other=combine_other)
        return cluster_proportions

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline

    def get_acls(self):
        acls, random_acls = get_acls(self.participant_strategies,
                                     self.pids, self.planning_data['envs'], self.planning_data['clicks'],
                                     self.pipeline, self.features, self.normalized_features, self.strategy_weights)
        return acls, random_acls

    def get_proportion_clusters(self, mode='participant', plot=True, show_clusters=False, n_clusters=2,
                                max_clusters=10):
        decision_proportions = []
        considered_pids = []
        for pid in self.pids:
            dp = self.participants[pid].decision_system_proportions
            if dp.shape[0] == self.num_trials:
                considered_pids.append(pid)
                decision_proportions.append(dp)
        if mode == 'participant':
            decision_proportions = np.mean(decision_proportions, axis=1)
        elif mode == "time":
            decision_proportions = np.mean(decision_proportions, axis=0)
        errors = []
        n_samples = decision_proportions.shape[0]
        if n_samples < max_clusters:
            max_clusters = n_samples
        for k in range(2, max_clusters):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(decision_proportions)
            errors.append(kmeans.inertia_)
        if plot:
            plt.plot(range(2, len(errors) + 2), errors)
            plt.xlabel("Number of clusters")
            plt.ylabel("Error (Inertia)")
            # plt.show()
        if n_clusters:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(decision_proportions)
            labels = kmeans.labels_
            cluster_map = {}
            cluster_dict = defaultdict(list)
            if mode == 'participant':
                for p, l in zip(considered_pids, labels):
                    cluster_map[p] = l
                    cluster_dict[l].append(p)
                if show_clusters:
                    for l, v in cluster_dict.items():
                        print(l)
                        for p in v:
                            print(self.participant_strategies[p])
            elif mode == "time":
                for i, l in enumerate(labels):
                    cluster_map[i + 1] = l
                    cluster_dict[l].append(i + 1)
                if show_clusters:
                    for l, v in cluster_dict.items():
                        print(l)
                        for t in v:
                            S = []
                            for pid in considered_pids:
                                S.append(self.participant_strategies[pid][t - 1])
                            print(t, S)
            cluster_dict = dict(cluster_dict)
            return cluster_dict

    def get_top_k_strategies(self, k=3):
        """
        Get the top k strategies from each trial.
        Example: 35 trials, look at each trial and get the top k strategies in each trial and put them together in a set
        Args:
            k: number of strategies

        Returns:

        """
        trial_wise_strategy_proportions = self.get_strategy_proportions(trial_wise=True)
        total_set = set()
        for t in trial_wise_strategy_proportions.keys():
            sorted_indices = sorted(trial_wise_strategy_proportions[t].items(), key=operator.itemgetter(1),
                                    reverse=True)[:k]
            for s, v in sorted_indices:
                if v > 0:
                    total_set.add(s)
        S = list(total_set)
        return S

    def plot_adaptive_maladaptive_strategies_vs_rest(self, adaptive_strategy_list, maladaptive_strategy_list):
        """
        This function sums up the proportion of the top 3 adaptive strategies and worst 3 maladaptive strategies and
        plots them against the summed proportions of the rest

        Args:
            adaptive_strategy_list: Manually defined list of adaptive (good) strategies
            maladaptive_strategy_list: Manually defined list of maladaptive (bad) strategies

        Returns: None

        """
        number_of_trials = list(range(0, self.num_trials))
        df = pd.DataFrame(float(0), index=number_of_trials, columns=["adaptive_strategy_sum", "maladaptive_strategy_sum", "rest"])
        for trial_key, strategy_dict in self.trial_strategy_proportions.items():
            for strategy_number, strategy_value in strategy_dict.items():
                if strategy_number in adaptive_strategy_list:
                    df["adaptive_strategy_sum"][trial_key] += strategy_value
                elif strategy_number in maladaptive_strategy_list:
                    df["maladaptive_strategy_sum"][trial_key] += strategy_value
                else:
                    df["rest"][trial_key] += strategy_value

        fig = plt.figure(figsize=(15, 10))

        plt.plot(range(1, self.num_trials + 1), df["adaptive_strategy_sum"], label="Adaptive strategies",
                 linewidth=3.0)
        plt.plot(range(1, self.num_trials + 1), df["maladaptive_strategy_sum"], label="Maladaptive strategies",
                 linewidth=3.0)
        plt.plot(range(1, self.num_trials + 1), df["rest"], label="Other strategies", linewidth=3.0)

        plt.xlabel("Trial Number", fontsize=24)
        plt.ylabel("Proportion", fontsize=24)
        # plt.title(title, fontsize=24)
        plt.ylim(top=1.0)
        plt.tick_params(labelsize=22)
        plt.legend(prop={'size': 23}, ncol=3, loc='upper center')
        plt.savefig(
            f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_aggregated_adaptive_maladaptive_other_strategies.png",
            dpi=400, bbox_inches='tight')
        plt.close(fig)
        return df["adaptive_strategy_sum"], df["maladaptive_strategy_sum"]

    def plot_decision_systems_proportions_intotal(self, DS_proportions, plot=True):
        decision_system_labels = ["Mental effort avoidance", "Model-based Metareasoning",
                                  "Model-free values and heuristics",
                                  "Pavlovian", "Satisficing and stopping"]
        data_columns = ['Experiment', 'Trial', 'Decision System', 'Relative Influence (%)']

        def get_ds_data(strategies, experiment_num):
            data = []
            for pid in strategies.keys():
                for i in range(len(strategies[pid])):
                    for j in range(len(decision_system_labels)):
                        data.append([experiment_num, i, decision_system_labels[j],
                                     DS_proportions[strategies[pid][i] - 1][j] * 100])
            return data

        data = get_ds_data(self.participant_strategies, self.exp_num)
        df = pd.DataFrame(data, columns=data_columns)
        if plot:
            fig = plt.figure(figsize=(15, 9))
            # plt.ylim(top=60)
            sns.barplot(x="Experiment", y="Relative Influence (%)", hue="Decision System", data=df)
            # plt.show()
            plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/decision_systen_proportion_total.png",
                        bbox_inches='tight')
            plt.close(fig)
        else:
            # averaged_df = df.groupby('Decision System').mean() # this does not work because number of participants need to be set manually
            aggregated_df = df.groupby('Decision System').sum()
            averaged_df = (aggregated_df / 35) / 15  # divided by number of participants and trials
            return averaged_df

    def plot_strategies_proportions_intotal(self):
        reward_structures_count = self.get_strategy_proportions()  # porportion of strategies

        strategies_set = self.get_top_k_strategies(k=3)
        strategies_list = sorted(list(strategies_set))  # get top k strategies

        data = []
        columns = ['Experiment', 'Strategy', 'Proportion (%)']
        # todo: does this make sense?
        strategy_labels = ["Random search for best possible final outcome", "Myopic Forward planning with satisficing",
                           "No planning", "Satisficing Best First Search", "Excessive goal-setting",
                           "Some immediate outcomes after all final outcomes",
                           "Immediate and final outcomes with satisficing",
                           "Intermediate outcome of the best immediate outcome"]

        for strategy in strategies_list:
            # data.append([reward_structures, strategy_labels[strategies_list.index(strategy)], reward_structures[strategy]*100])
            data.append([self.exp_num, strategy, reward_structures_count[strategy] * 100])
        df = pd.DataFrame(data, columns=columns)
        plt.figure(figsize=(12, 9))
        sns.barplot(x='Experiment', y='Proportion (%)', hue='Strategy', data=df)
        # plt.show()
        plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/strategy_proportion_total.png", bbox_inches='tight')

    def plot_clusters_proportions_intotal(self):
        reward_structures_count = self.get_cluster_proportions()
        data = []
        columns = ['Experiment', 'Cluster Type', 'Proportion (%)']
        # cluster_labels = ["Immediate outcomes of the best final outcomes", "Local Search", "Frugal planning", "Maximizing goal-setting without backward planning", "Other Goal setting strategies", "Miscellaneous strategies"]

        # t_prop = 0
        for keys, values in reward_structures_count.items():
            data.append([self.exp_num, keys, values * 100])
            # t_prop += reward_structures_count[cluster]
        df = pd.DataFrame(data, columns=columns)

        plt.figure(figsize=(12, 9))
        sns.barplot(x='Experiment', y='Proportion (%)', hue='Cluster Type',
                    data=df)  # todo: add actual numbers to the plot
        plt.ylim(top=60)
        # plt.show()
        plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/cluster_proportion_total.png", bbox_inches='tight')

    ### All about changes ###
    def trial_decision_system_change_rate(self, decision_system_by_trial):
        difference = np.diff(decision_system_by_trial, axis=0)
        # difference_sum = np.sum(difference, axis=1)
        decision_system_labels = ["Mental effort avoidance", "Model-based Metareasoning",
                                  "Model-free values and heuristics",
                                  "Pavlovian", "Satisficing and stopping"]
        fig = plt.figure(figsize=(15, 10))
        prefix = "Decision System"
        for i in range(difference.shape[1]):
            plt.plot(range(1, difference.shape[0] + 1), difference[:, i], label=decision_system_labels[i],
                     linewidth=3.0)
        plt.xlabel("Trial Number", fontsize=24)
        plt.ylabel("Rate of change of decision systems", fontsize=24)
        # plt.title(title, fontsize=24)
        plt.ylim(top=0.2)
        plt.tick_params(labelsize=22)
        plt.legend(prop={'size': 23}, ncol=3, loc='upper center')
        plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_decision_system_change_rate.png",
                    dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    def trial_cluster_change_rate(self, trial_prop, C):
        C_proportions = []
        for t in trial_prop.keys():
            props = []
            for clusters in C:
                props.append(trial_prop[t].get(clusters, 0))
            C_proportions.append(props)
        C_proportions = np.array(C_proportions)
        difference = np.diff(C_proportions, axis=0)
        fig = plt.figure(figsize=(15, 10))
        prefix = "Cluster"
        for i in range(difference.shape[1]):
            plt.plot(range(1, difference.shape[0] + 1), difference[:, i], label=f"{prefix} {i + 1}", linewidth=3.0)
        plt.xlabel("Trial Number", fontsize=24)
        plt.ylabel("Rate of change of clusters", fontsize=24)
        # plt.title(title, fontsize=24)
        plt.ylim(top=0.4)
        plt.tick_params(labelsize=22)
        plt.legend(prop={'size': 23}, ncol=3, loc='upper center')
        plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/{self.exp_num}_cluster_change_rate.png",
                    dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    def analyze_trajectory(self, trajectory, print_trajectories=False):
        final_repetition_count = []
        for tr in trajectory:
            if len(tr[0]) > 1:
                if print_trajectories:
                    print("Trajectory:", tr[0][0])
                    print("Repetition Frequency:", tr[0][1])
                    # print("Freq:", tr[1], "\n")
                temp = list(tr[0][1])
                temp.pop()
                number_of_trials_before_last_trial = np.sum(temp)
                # final_repetition_count.append(tr[0][1][-1])
                final_repetition_count.append(number_of_trials_before_last_trial)
                # print("The last item in Repetition Frequency", tr[0][1][-1])

        average_trials_repetition = np.mean(final_repetition_count)
        median_trials_repetition = np.median(final_repetition_count)
        print("Median final strategy usage: ", median_trials_repetition)
        print("Mean final strategy usage:", average_trials_repetition)


    def remove_duplicates(self, cluster_list): #this one should go into utils
        previous_cluster = cluster_list[0]
        non_duplicate_list = [previous_cluster]
        duplicate_freqs = [1]
        for i in range(1, len(cluster_list)):
            if cluster_list[i] != previous_cluster:
                non_duplicate_list.append(cluster_list[i])
                previous_cluster = cluster_list[i]
                duplicate_freqs.append(1)
            else:
                duplicate_freqs[-1] += 1
        res = (tuple(non_duplicate_list), tuple(duplicate_freqs))
        return res

    def get_sorted_trajectories(self, cluster_map, strategies):
        """
        Assign frequency to cluster and strategies
        Args:
            strategies: load from strategies.pkl

        Returns: cluster and strategy list in list, where first list describes the cluster/strategy and second list describes
        the frequency. Example: [((39, 37, 61, 39, 37, 39, 37, 39, 33, 49, 39), (1, 2, 1, 3, 4, 1, 4, 5, 1, 1, 12)), 1]
        Reads: strategy 39 was used for 1 trials, then strategy 37 was used for 2 trials

        """
        cluster_trajectory_frequency = defaultdict(int)
        strategy_trajectory_frequency = defaultdict(int)
        for pid, strategy_sequence in strategies.items():
            cluster_strategy_sequence = [cluster_map[strategy] for strategy in strategy_sequence]
            cluster_trajectory = self.remove_duplicates(cluster_strategy_sequence)
            strategy_trajectory = self.remove_duplicates(strategy_sequence)
            cluster_trajectory_frequency[cluster_trajectory] += 1
            strategy_trajectory_frequency[strategy_trajectory] += 1
        sorted_cluster_trajectory = [list(s) for s in
                                     sorted(cluster_trajectory_frequency.items(), key=operator.itemgetter(1), reverse=True)]
        sorted_strategy_trajectory = [list(s) for s in
                                      sorted(strategy_trajectory_frequency.items(), key=operator.itemgetter(1),
                                             reverse=True)]
        return sorted_cluster_trajectory, sorted_strategy_trajectory

    def plot_difference_between_trials(self, cluster_map, strategies: defaultdict, number_participants, cluster=False):
        """
        It creates a plot which shows the percentage of participants who changed their strategy across trial
        Args:
            strategies: A list of strategies for all participants (index is pid) across trials.
            number_participants: fixed number of participants

        Returns: two plots, one that plots percentage of participants that changed their strategy and strategy cluster

        """
        change_list_of_dicts = []
        for key, value in strategies.items():
            if cluster:
                # mapping strategy to cluster
                value = [cluster_map[strategy] for strategy in value]

            changes_numeric = np.diff(value)
            # Convert result of numpy difference into dictionary that maps trial_index -> whether a change occurred (1 or 0)
            change_count = {trial_idx: int(diff_val != 0) for trial_idx, diff_val in enumerate(list(changes_numeric))}
            change_list_of_dicts.append(change_count)  # a dict of all changes for each participant, len: 15

        df = pd.DataFrame(change_list_of_dicts)
        sum_values = df.sum(axis=0)

        fig = plt.figure(figsize=(15, 10))
        # create percentages by dividing each item in the list by number of participants (15)
        relative_sum_values = [x / number_participants for x in list(sum_values)]
        if cluster:
            plt.bar(sum_values.keys(), relative_sum_values, 1, color='b')
            plt.ylim(top=1.0)
            plt.xlabel("Trial Number", size=24)
            plt.ylabel("Percentage of people who changed strategy cluster", fontsize=24)
            plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/absolute_number_of_changes_cluster.png",
                        bbox_inches='tight')
        else:
            plt.bar(sum_values.keys(), relative_sum_values, 1, color='b')
            plt.ylim(top=1.0)
            plt.xlabel("Trial Number", size=24)
            plt.ylabel("Percentage of people who changed strategy", fontsize=24)
            plt.savefig(f"../results/cm/plots/{self.exp_num}_{self.block}/absolute_number_of_changes_strategy.png",
                        bbox_inches='tight')
        plt.close(fig)
        return None

    def analysis_change_percentage(self, precomputed_strategies, cluster_map):

        number_participants = len(self.participants)

        # clusters = learning_utils.pickle_load("data/kl_clusters.pkl")
        self.plot_difference_between_trials(cluster_map, precomputed_strategies, number_participants, cluster=False)
        self.plot_difference_between_trials(cluster_map, precomputed_strategies, number_participants, cluster=True)

        # Get sorted trajectories
        cluster_trajectory, strategy_trajectory = self.get_sorted_trajectories(cluster_map, precomputed_strategies)

        # show how many trials until the final strategy was used
        print("Strategy usage:")
        self.analyze_trajectory(strategy_trajectory, print_trajectories=False)
        print("\n")

        # show how many trials until the final strategy cluster was used
        print("Cluster usage:")
        self.analyze_trajectory(cluster_trajectory, print_trajectories=True)
        print("\n")


    def summarize(self, features, normalized_features, strategy_weights,
                  decision_systems, W_DS,
                  DS_proportions, strategy_scores, cluster_scores, cluster_map,
                  manual_strategy_list=[], maladaptive_strategy_list=[],
                  max_evals=20,
                  plot_strategies=[21, 30], plot_clusters=list(range(1, 14)),
                  n_clusters=None, max_clusters=10,
                  cluster_mode="participant",  # Can also take time,
                  show_pids=True,
                  show_strategies=False,
                  precomputed_strategies=None,
                  precomputed_temperatures=None,
                  ):
        """
        Creates plots about 1. strategy development over trials and overall frequency, 2. strategy cluster development over trials and overall frequency,
        3. decision system development over trials and overall frequency (6 plots in total).
        The plots can be found in the results folder
        Args:
            features:
            normalized_features:
            strategy_weights:
            decision_systems:
            W_DS:
            DS_proportions:
            strategy_scores:
            cluster_scores:
            cluster_map:
            max_evals:
            plot_strategies:
            plot_clusters:
            n_clusters:
            max_clusters:
            cluster_mode:
            show_pids:
            show_strategies:
            precomputed_strategies:
            precomputed_temperatures:

        Returns:

        """
        self.infer_strategies(precomputed_strategies=precomputed_strategies,
                              precomputed_temperatures=precomputed_temperatures,
                              max_evals=max_evals, show_pids=show_pids)
        if show_strategies:
            print("\n", dict(self.participant_strategies), "\n")
        self.init_feature_properties(features, normalized_features, strategy_weights)
        self.init_decision_system_properties(decision_systems, W_DS, DS_proportions)
        # clusters = self.get_proportion_clusters(mode=cluster_mode, show_clusters=True, plot=True,
        #                                         n_clusters=n_clusters, max_clusters=max_clusters)
        # print("Clusters:", clusters, "\n")
        self.pipeline = self.cm.pipeline
        # acls, random_acls = self.get_acls()
        # mean_acl = np.mean(acls)
        # mean_random_acl = np.mean(random_acls)
        # print("ACL:", mean_acl, "ACL-Random:", mean_random_acl, "\n")
        # print(f"ACL factor: {mean_acl / mean_random_acl}", "\n")

        self.strategy_transitions_chi2()
        self.performance_transitions_chi2(strategy_scores=strategy_scores)
        self.frequency_transitions_chi2()

        self.init_strategy_clusters(cluster_map)
        self.strategy_transitions_chi2(clusters=True)
        self.performance_transitions_chi2(cluster_scores=cluster_scores)
        self.frequency_transitions_chi2(clusters=True)

        # plot regarding strategy clusters
        self.plot_cluster_proportions(C=plot_clusters)
        self.trial_cluster_change_rate(self.trial_cluster_proportions, C=plot_clusters)
        self.plot_clusters_proportions_intotal()

        # plot regarding decision systems
        mean_dsw = self.plot_average_ds()
        self.trial_decision_system_change_rate(mean_dsw)
        self.plot_decision_systems_proportions_intotal(DS_proportions, plot=True)

        # plot regarding the strategies
        S = self.get_top_k_strategies(k=3)
        self.plot_strategy_proportions_pertrial(S)
        self.plot_strategies_proportions_intotal()
        self.plot_strategy_scores(strategy_scores)  # not saved as plot
        if manual_strategy_list:
            self.plot_adaptive_maladaptive_strategies_vs_rest(manual_strategy_list, maladaptive_strategy_list)

        # plot regarding the change between trials
        self.analysis_change_percentage(precomputed_strategies, cluster_map)
        # self.plot_parallel_coordinates(mode=cluster_mode)

    def statistical_kpis(self, features, normalized_features, strategy_weights,
                         decision_systems, W_DS,
                         DS_proportions, strategy_scores, cluster_scores, cluster_map,
                         max_evals=20,
                         show_pids=True,
                         show_strategies=False,
                         precomputed_strategies=None,
                         precomputed_temperatures=None
                         ):

        self.infer_strategies(precomputed_strategies=precomputed_strategies,
                              precomputed_temperatures=precomputed_temperatures,
                              max_evals=max_evals, show_pids=show_pids)
        if show_strategies:
            print("\n", dict(self.participant_strategies), "\n")
        self.init_feature_properties(features, normalized_features, strategy_weights)
        self.init_decision_system_properties(decision_systems, W_DS, DS_proportions)

        self.pipeline = self.cm.pipeline

        # self.strategy_transitions_chi2()
        # self.performance_transitions_chi2(strategy_scores=strategy_scores)
        # self.frequency_transitions_chi2()'

        self.init_strategy_clusters(cluster_map)
        # self.strategy_transitions_chi2(clusters=True)
        # self.performance_transitions_chi2(cluster_scores=cluster_scores)
        # self.frequency_transitions_chi2(clusters=True)

        # strategies
        # S = self.get_top_k_strategies(k=3)
        strategy_proportions = self.get_strategy_proportions()
        strategy_proportions_trialwise = self.get_strategy_proportions(trial_wise=True)

        # clusters
        cluster_proportions = self.get_cluster_proportions()
        cluster_proportions_trialwise = self.get_cluster_proportions(trial_wise=True)

        # decision systems
        decision_system_proportions = self.plot_decision_systems_proportions_intotal(DS_proportions, plot=False)
        mean_dsw = self.plot_average_ds()

        return strategy_proportions, strategy_proportions_trialwise, cluster_proportions, cluster_proportions_trialwise, decision_system_proportions, mean_dsw

    # todo: Clean repetitive code
