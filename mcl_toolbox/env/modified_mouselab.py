import random
from collections import defaultdict, OrderedDict
from functools import lru_cache
from statistics import mean
from typing import List

import numpy as np
from toolz import get

from mcl_toolbox.utils import distributions
from mcl_toolbox.utils.learning_utils import get_normalized_feature_values

""" This file defines the node, trial and trial sequence class for the
    feature based representation of the Mouselab-MDP. This assumes that
    the environment structure is a tree symmetric in its branches
"""
# distribution_by_level = {1: [-4, -2, 2, 4], 2: [-8, -4, 4, 8], 3: [-48, -24, 24, 48]}
distribution_by_level = {1: [-1, 1], 2: [-5], 3: [-5, -50, 50]}  # todo: does this make sense for strategy discovery?


# distribution_by_level = [[0], [-1, 1], [-5, 5], [-5, 5, -50, 50]]
# decreasing_dist_by_level = {1: [-48, -24, 24, 48], 2: [-8, -4, 4, 8], 3: [-4, -2, 2, 4]}

# constant_list = [-10, -5, 5, 10]

# constant_dist_by_level = {k: constant_list for k in range(5)}


def key_with_max_val(best_map):
    v = list(best_map.values())
    k = list(best_map.keys())
    return k[v.index(max(v))]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@lru_cache(maxsize=None)
def reward_val(depth):
    if depth > 0:
        return distributions.Categorical(distribution_by_level[depth])
    return 0.0


# todo: does this make sense for strategy discovery?

def constant_reward_val(depth):
    if depth > 0:
        return distributions.Categorical(constant_dist_by_level[depth])
    return 0.0


def decreasing_reward_val(depth):
    if depth > 0:
        return distributions.Categorical(decreasing_dist_by_level[depth])
    return 0.0


@lru_cache(maxsize=None)
def normal_reward_val(depth):
    def convert_dist(d):
        vals = d.vals
        probs = d.probs
        val_prob = defaultdict(int)
        for i, (v, p) in enumerate(zip(vals, probs)):
            val_prob[v] += p
        vals = sorted(list(set(vals)))
        new_dist = distributions.Categorical(
            vals=vals, probs=[val_prob[v] for v in vals]
        )
        return new_dist

    stds = [1, 2, 4, 8, 32]
    if depth > 0:
        dist = distributions.Normal(0, stds[depth - 1]).to_discrete(n=257, max_sigma=4)
        dist.vals = tuple([int(round(val)) for val in dist.vals])
        dist = convert_dist(dist)
        return dist
    return 0.0


def approx_max(dist, position=0):
    if hasattr(dist, "mu"):
        if position == 0:
            return dist.mu + 2 * dist.sigma
        else:
            return dist.mu + dist.sigma
    else:
        if position == 0:  # todo: position is always 0? check for non sd conditions
            return max(dist.vals)
        else:
            # return sorted(dist.vals)[-(position + 1)] #todo: I actually think this previous implementation is wrong as it chooses the second best one instead of the best one
            # return sorted(dist.vals)[-(position)] #I guess this is for choosing the largest option by sorting and then the last one in the que
            return sorted(dist.vals)[
                -1]  # I guess this is for choosing the largest option by sorting and then the last one in the que


def approx_min(dist, position=0):
    if hasattr(dist, "mu"):
        if position == 0:
            return dist.mu - 2 * dist.sigma
        else:
            return dist.mu - dist.sigma
    return sorted(dist.vals)[position]


def get_termination_mers(envs, trial_actions, pipeline):
    mers = []
    for i, (env, actions) in enumerate(zip(envs, trial_actions)):
        trial = TrialSequence(1, [pipeline[i]], ground_truth=[env]).trial_sequence[0]
        for action in actions[:-1]:
            trial.node_map[action].observe()
        mers.append(trial.node_map[1].calculate_max_expected_return())
    return mers


class TrialSequence:
    def __init__(
            self, num_trials: int, pipeline: dict, ground_truth: List[List[float]] = None
    ) -> None:
        self.num_trials = num_trials
        self.pipeline = pipeline
        if not ground_truth:
            self._construct_ground_truth()
        else:
            self.ground_truth = ground_truth
        if len(self.ground_truth) != num_trials:
            raise ValueError(
                "The length of ground truth must be equal to the number of trials"
            )
        for gt in self.ground_truth:
            gt[0] = 0.0
        self.trial_sequence = []
        self._construct_trials()
        # THINK ABOUT THIS AND FIX IT
        self.node_click_count = {i: 0 for i in range(100)}
        self.observed_node_values = defaultdict(list)

    def _construct_ground_truth(self):
        """
        Construct ground truth from reward distribution
        (Used Fred's code from mouselab_utils.py)
        """
        self.ground_truth = []

        def gen_trial_gt(trial_num):
            tree = []
            init = []
            branching = self.pipeline[trial_num][0]
            reward_function = self.pipeline[trial_num][1]

            def expand(d):
                nonlocal init, tree
                my_idx = len(init)
                init.append(reward_function(d))
                children = []
                tree.append(children)
                for _ in range(get(d, branching, 0)):  # get position d from list branching, if not found return 0
                    child_idx = expand(d + 1)
                    children.append(child_idx)
                return my_idx

            expand(0)
            dist = (0, *init[1:])
            return list(map(distributions.sample, dist))

        for trial_num in range(self.num_trials):
            gt = gen_trial_gt(trial_num)
            self.ground_truth.append(gt)

    def _construct_structure_map(self, branching):
        """
        Construct the structure map from which the trial representation will be created.
        Assumes symmetric structure.

        Returns:
                dict -- Keys are node numbers and parents are values.
        """
        structure_map = {0: None}
        branching_len = len(branching)
        curr_index = 1

        def construct_map(curr_parent, branching_index):
            if branching_index == branching_len:
                return
            nonlocal curr_index, structure_map
            for _ in range(branching[branching_index]):
                present_node = curr_index
                structure_map[curr_index] = curr_parent
                curr_index += 1
                construct_map(present_node, branching_index + 1)

        construct_map(0, 0)
        return structure_map

    def _construct_trials(self):
        values = self.ground_truth
        for trial_num in range(self.num_trials):
            branching = self.pipeline[trial_num][0]
            reward_function = self.pipeline[trial_num][1]
            structure_map = self._construct_structure_map(branching)
            trial = Trial(
                values[trial_num],
                structure_map,
                max_depth=len(branching),
                reward_function=reward_function,
            )
            trial.sequence = self
            self.trial_sequence.append(trial)

    def increment_count(self, click):
        self.node_click_count[click] += 1

    def decrement_count(self, click):
        self.node_click_count[click] -= 1

    def get_click_count(self, click):
        return self.node_click_count[click]

    def reset_count(self):
        for k in self.node_click_count.keys():
            self.node_click_count[k] = 0

    def reset_sequence(self):
        self._construct_trials()
        self.reset_count()
        self.observed_node_values = defaultdict(list)


class Trial:
    def __init__(
            self, ground_truth, structure_map, max_depth=None, reward_function=None
    ):
        self.sequence = None
        self.construct_trial(ground_truth, structure_map)
        self.previous_observed = None
        if max_depth:
            self.max_depth = max_depth
        if reward_function:
            self.reward_function = reward_function
        self.node_level_map = {}
        self.level_map = {d: [] for d in range(1, self.max_depth + 1)}
        self.construct_node_maps(self.root, 0)
        self.init_expectations()
        self.path_map = {}
        self.branch_map = {}  # Nodes lying on the branch
        self.reverse_branch_map = {}  # Branches passing through the node
        self.construct_branch_maps()
        self.observed_nodes = []
        self.unobserved_nodes = list(self.node_map.values())
        self.num_nodes = len(self.node_map)
        self.max_values_by_depth = {
            # d: approx_max(self.reward_function(d)) for d in range(1, self.max_depth + 1)
            d: approx_max(self.reward_function(d), position=d) for d in range(1, self.max_depth + 1)
        }
        self.min_values_by_depth = {
            d: approx_min(self.reward_function(d)) for d in range(1, self.max_depth + 1)
        }
        self.variance_by_depth = {
            d: self.reward_function(d).var() for d in range(1, self.max_depth + 1)
        }
        self.uncertainty_by_depth = {
            d: self.reward_function(d).std() for d in range(1, self.max_depth + 1)
        }

    def create_node_label(self, label, parent):
        node = Node(self)
        node.label = label
        if parent is not None:
            parent.children.append(node)
            node.parent = parent
            node.root = parent.root
        else:
            node.root = node
        node.tree = self
        return node

    def construct_trial(self, ground_truth, parent_map):
        node_map = {}
        ground_truth[0] = 0.0
        for k, v in parent_map.items():
            if v is not None:
                node_map[k] = self.create_node_label(k, node_map[v])
            else:
                node_map[k] = self.create_node_label(0, None)
            node_map[k].value = ground_truth[k]
        self.node_map = node_map
        self.root = node_map[0]
        self.ground_truth = ground_truth

    def construct_node_maps(self, root, current_level):
        for child in root.children:
            self.level_map[current_level + 1].append(child)
            self.node_level_map[child.label] = current_level + 1
            child.depth = current_level + 1
        for child in root.children:
            self.construct_node_maps(child, current_level + 1)

    def init_expectations(self):
        for node in self.node_map.values():
            if node.label != 0:
                node.expected_value = self.reward_function(node.depth).expectation()

    def construct_branch_maps(self):
        path_num = 1

        def get_tree_path(root, present_path):
            nonlocal path_num
            present_path.append(root.label)
            if not root.children:
                self.branch_map[path_num] = tuple(present_path)
                for node_num in present_path:
                    if node_num not in self.reverse_branch_map:
                        self.reverse_branch_map[node_num] = [path_num]
                    else:
                        self.reverse_branch_map[node_num].append(path_num)
                present_path = []
                path_num += 1
            else:
                for child in root.children:
                    get_tree_path(child, present_path[:])

        get_tree_path(self.root, [])

    def reset_observations(self):
        num_nodes = len(self.node_map)
        for node_num in range(num_nodes):
            self.node_map[node_num].observed = False
        self.previous_observed = None
        self.observed_nodes = []
        self.unobserved_nodes = list(self.node_map.values())

    def get_action_feedback(self, taken_path):
        path_sums = {}
        for branch in range(1, len(self.branch_map) + 1):
            path_sums[branch] = sum(
                [self.node_map[node].value for node in self.branch_map[branch]]
            )
        max_branch_sum = max(path_sums.values())
        taken_branches = self.reverse_branch_map[taken_path[1]]
        max_taken_sum = max([path_sums[branch] for branch in taken_branches])
        if max_branch_sum == max_taken_sum:
            return 0
        else:
            return 2 + max_branch_sum - max_taken_sum

    def get_node_feature_values(self, nodes, features, normalized_features):
        num_features = len(features)
        feature_values = np.zeros((len(nodes), num_features))
        for i, node in enumerate(nodes):
            # for each node, get feature_values as list
            feature_values[i] = list(node.compute_termination_feature_values(features).values())
            if normalized_features:
                feature_values[i] = get_normalized_feature_values(
                    feature_values[i], features, normalized_features
                )
        return feature_values

    def get_leaf_nodes(self):
        leaf_nodes = []
        for node in self.node_map.values():
            if node.is_leaf():
                leaf_nodes.append(node)
        return leaf_nodes

    def set_previous_node(self, node):
        self.previous_observed = node

    def unobserve(self, node_num):
        self.node_map[node_num].observed = False
        node = self.node_map[node_num]
        self.observed_nodes.remove(node)
        self.unobserved_nodes.append(node)
        self.previous_observed = None
        if len(self.observed_nodes) > 0:
            self.previous_observed = self.observed_nodes[-1]
        self.sequence.decrement_count(node_num)
        self.sequence.observed_node_values[self.node_level_map[node_num]].remove(
            node.value
        )

    def get_max_dist_value(self):
        max_values = list(self.max_values_by_depth.values())
        max_value = max(max_values)
        return max_value

    def get_min_dist_value(self):
        min_values = list(self.min_values_by_depth.values())
        min_value = min(min_values)
        return min_value

    def get_observed_nodes(self):
        return self.observed_nodes

    def get_observed_node_count(self):
        return len(self.observed_nodes)

    def get_unobserved_nodes(self):
        return self.unobserved_nodes

    def get_unobserved_node_count(self):
        return len(self.unobserved_nodes)

    def get_num_clicks(self):
        return len(self.observed_nodes)

    def immediate_termination(self):
        if not self.previous_observed:
            return -1
        return 0

    def first_node_observed(self):
        if self.previous_observed:
            return -1
        return 0

    def is_positive_observed(self):
        for node in self.observed_nodes:
            if node.value > 0:
                return -1
        return 0

    def all_roots_observed(self):
        node_list = self.level_map[1]
        for node in node_list:
            if not node.observed:
                return 0
        return -1

    def all_leaf_nodes_observed(self):
        node_list = self.level_map[self.max_depth]
        for node in node_list:
            if not node.observed:
                return 0
        return -1

    def positive_root_leaves_termination(self):
        root_nodes = self.level_map[1]
        pos_node_list = []
        for node in root_nodes:
            if node.observed and node.value > 0:
                pos_node_list.append(node)
        if not pos_node_list:
            return 0
        leaf_nodes = self.level_map[self.max_depth]
        for node in leaf_nodes:
            present_node = node
            while present_node.parent.label != 0:
                present_node = present_node.parent
            if present_node in pos_node_list:
                if not node.observed:
                    return 0
        return -1

    def single_path_completion_termination(self):
        leaf_nodes = self.level_map[self.max_depth]
        observed_leaf_nodes = []
        for node in leaf_nodes:
            if node.observed:
                observed_leaf_nodes.append(node)
        if not observed_leaf_nodes:
            return 0
        for node in observed_leaf_nodes:
            present_node = node
            path_complete = True
            while present_node.parent.label != 0:
                if not present_node.parent.observed:
                    path_complete = False
                    break
                present_node = present_node.parent
            if path_complete:
                return -1
        return 0

    def is_previous_max(self):
        max_value = self.get_max_dist_value()
        if self.previous_observed:
            if self.previous_observed.value >= max_value:
                return -1
        return 0

    def hard_satisficing(self, aspiration_level):
        expected_path_values = self.get_path_expected_values()
        max_expected_value = max(expected_path_values.values())
        if max_expected_value >= aspiration_level:
            return 0
        return -1

    def soft_satisficing(self):
        return 0

    def largest_value_observed(self):
        num_branches = len(self.branch_map)
        max_path_values = {}
        for i in range(1, num_branches + 1):
            path = self.branch_map[i]
            max_value = -9999
            for node_num in path:
                node = self.node_map[node_num]
                if node.observed:
                    max_value = max(max_value, node.value)
            if max_value == -9999:
                max_value = 0
            max_path_values[i] = max_value
        return max_path_values

    def get_path_expected_values(self):
        """
        Loops through all branches and get expected value of each branch /path
        Returns: a dict that contains the path and its expected value

        """
        # self.branch_map is a dict containing the branches and indeces,
        # e.g. for 3 step task {1: (0, 1, 2, 3), 2: (0, 1, 2, 4), 3: (0, 5, 6, 7), 4: (0, 5, 6, 8), 5: (0, 9, 10, 11), 6: (0, 9, 10, 12)}
        num_branches = len(self.branch_map)
        expected_path_values = {}
        for i in range(1, num_branches + 1):
            path = self.branch_map[i]
            ev = 0
            for node_num in path:
                node = self.node_map[node_num]  # node object
                if node.observed:
                    ev += node.value
                else:
                    # if not the starting node
                    if node_num != 0:
                        # Verify this
                        ev += node.expected_value
                        # given our symmetric setup, expectation is always 0
                        # expected value is calculated by taking the depths and selecting the reward function e.g.
                        # depth 1 = (-4, -2, 2, 4) with probs [0.25, 0.25, 0.25, 0.25]
            expected_path_values[i] = ev
        return expected_path_values

    def get_path_expected_values_information(self, level_values):
        num_branches = len(self.branch_map)
        expected_path_values = {}
        for i in range(1, num_branches + 1):
            path = self.branch_map[i]
            ev = 0
            for node_num in path:
                node = self.node_map[node_num]
                if node.observed:
                    ev += node.value
                else:
                    # Replacing depth_by_num[node.label] with node.depth
                    ev += level_values[node.depth]
            expected_path_values[i] = ev
        return expected_path_values

    def get_improvement_expected_values(self):
        num_branches = len(self.branch_map)
        best_paths = []
        other_paths = []
        for i in range(1, num_branches + 1):
            flag = 0
            path = self.branch_map[i]
            ev = 0
            for node_num in path:
                node = self.node_map[node_num]
                if node.on_most_promising_path() == 1 and node.label != 0:
                    flag = 1
                if node.observed:
                    ev += node.value
                elif node.label != 0:
                    if node.on_most_promising_path():
                        ev += self.min_values_by_depth[node.depth]
                    else:
                        ev += self.max_values_by_depth[node.depth]
            if flag == 0:
                other_paths.append(ev)
            else:
                best_paths.append(ev)
        return other_paths, best_paths

    def is_max_path_observed(self):
        observed_nodes = self.get_observed_nodes()
        max_value = self.get_max_dist_value()
        for node in observed_nodes:
            if node.value >= max_value:
                if node.check_path_observed():
                    return -1
        return 0

    def are_max_paths_observed(self):
        observed_nodes = self.get_observed_nodes()
        max_value = self.get_max_dist_value()
        if self.all_leaf_nodes_observed() == -1:
            for node in observed_nodes:
                if node.value >= max_value and not node.check_path_observed():
                    return 0
            return -1
        return 0

    def termination_leaves_observed(self):
        node_list = self.level_map[self.max_depth]
        for node in node_list:
            if not node.observed:
                return -1
        return 0

    def termination_positive(self):
        observed_nodes = self.get_observed_nodes()
        for node in observed_nodes:
            if node.value > 0:
                return 0
        return -1

    def termination_roots_observed(self):
        node_list = self.level_map[1]
        for node in node_list:
            if not node.observed:
                return -1
        return 0

    # negative value if it should not terminate and 0 if it should
    # swapping signs is okay, the weights can be adjusted

    def termination_first_node(self):
        if self.previous_observed:
            return 0
        return -1

    def termination_positive_root_leaves(self):
        root_nodes = self.level_map[1]
        pos_node_list = []
        for node in root_nodes:
            if node.observed and node.value > 0:
                pos_node_list.append(node)
        if not pos_node_list:
            return -1
        leaf_nodes = self.level_map[self.max_depth]
        for node in leaf_nodes:
            present_node = node
            while present_node.parent.label != 0:
                present_node = present_node.parent
            if present_node in pos_node_list:
                if not node.observed:
                    return -1
        return 0

    def termination_single_path(self):
        leaf_nodes = self.level_map[self.max_depth]
        observed_leaf_nodes = []
        for node in leaf_nodes:
            if node.observed:
                observed_leaf_nodes.append(node)
        if not observed_leaf_nodes:
            return -1
        for node in observed_leaf_nodes:
            present_node = node
            path_complete = True
            while present_node.parent.label != 0:
                if not present_node.parent.observed:
                    path_complete = False
                    break
                present_node = present_node.parent
            if path_complete:
                return 0
        return -1

    def termination_previous_max(self):
        max_value = self.get_max_dist_value()
        if self.previous_observed and self.previous_observed.value >= max_value:
            return 0
        return -1

    def termination_max_observed(self):
        observed_nodes = self.get_observed_nodes()
        max_value = self.get_max_dist_value()
        for node in observed_nodes:
            if node.value >= max_value:
                if node.check_path_observed():
                    return 0
        return -1

    def termination_max_paths_observed(self):
        observed_nodes = self.get_observed_nodes()
        max_value = self.get_max_dist_value()
        if self.all_leaf_nodes_observed() == 0:
            for node in observed_nodes:
                if node.value >= max_value and not node.check_path_observed():
                    return -1
            return 0
        return -1

    def calculate_expected_value_path(self, root, present_path, path_values):
        present_path_copy = present_path[:]
        path_values_copy = path_values[:]
        if root.parent is None:
            present_path_copy.append(0)
        else:
            present_path_copy.append(root.label)
        if root.observed:
            path_values_copy.append(root.value)
        else:
            path_values_copy.append(0)
        self.path_map[tuple(present_path_copy)] = sum(path_values_copy)
        for child in root.children:
            self.calculate_expected_value_path(
                child, present_path_copy, path_values_copy
            )

    def get_best_expected_path(self):
        self.calculate_expected_value_path(self.root, [], [])
        paths = {k: v for k, v in self.path_map.items() if len(k) == self.max_depth + 1}
        max_path_value = -999
        for k, v in paths.items():
            if v > max_path_value:
                max_path_value = v
        path_list = []
        for k, v in paths.items():
            if v == max_path_value:
                path_list.append(list(k))
        return random.choice(path_list)

    def get_random_path(self):
        self.calculate_expected_value_path(self.root, [], [])
        complete_paths = []
        for k, v in self.path_map.items():
            if len(k) == self.max_depth + 1:
                complete_paths.append([k, v])
        return random.choice(complete_paths[0])

    def get_termination_data(self):
        taken_path = []
        best_expected_path = self.get_best_expected_path()
        reward = 0
        for node in best_expected_path:
            taken_path.append(node)
            reward += self.node_map[node].value
        return taken_path, reward

    def termination_after_observing_positive_inner_and_one_outer(self):
        root_nodes = self.level_map[1]
        leaf_nodes = self.level_map[self.max_depth]
        observed_nodes = self.get_observed_nodes()

        leaf_count = sum(1 for node in observed_nodes if node in leaf_nodes)
        root_count = sum(1 for node in observed_nodes if node in root_nodes)

        # make sure that there is only 1 leaf node observed and all other observed nodes are level 1 nodes
        if leaf_count == 1 and root_count == len(observed_nodes) - 1:
            # is the last observed node a leaf: is the order correct?
            if observed_nodes[-1] in leaf_nodes:
                return 1
        return -1


class Node:
    def __init__(self, trial):

        self.observed = False
        self.value = None
        self.parent = None
        self.children = []
        self.depth = 0  # Will be initialized when a trial is created
        self.label = 0
        self.trial = trial
        self.feature_function_map = OrderedDict({
            "siblings_count": self.get_observed_siblings_count,
            "depth_count": self.get_observed_same_depth_count,
            "ancestor_count": self.get_observed_ancestor_count,
            "successor_count": self.get_observed_successor_count,
            "is_leaf": self.is_leaf,
            "depth": self.get_depth_node,
            "max_successor": self.get_max_successor_value,
            "parent_value": self.get_parent_value,
            "max_immediate_successor": self.get_max_immediate_successor,
            "immediate_successor_count": self.get_immediate_successor_count,
            "previous_observed_successor": self.is_previous_successor,
            "is_successor_highest": self.is_successor_highest_leaf,
            "parent_observed": self.is_parent_observed,
            "is_max_path_observed": self.trial.is_max_path_observed,
            "observed_height": self.get_observed_height,
            "are_max_paths_observed": self.trial.are_max_paths_observed,
            "is_previous_max": self.trial.is_previous_max,
            "is_positive_observed": self.trial.is_positive_observed,
            "all_roots_observed": self.trial.all_roots_observed,
            "all_leaf_nodes_observed": self.trial.all_leaf_nodes_observed,
            "immediate_termination": self.trial.immediate_termination,
            "is_pos_ancestor_leaf": self.is_leaf_and_positive_ancestor,
            "positive_root_leaves_termination": self.trial.positive_root_leaves_termination,
            "single_path_completion": self.trial.single_path_completion_termination,
            "is_root": self.is_root,
            "is_previous_successor_negative": self.is_previous_observed_successor_negative,
            "sq_successor_count": self.sq_successor_count,
            "uncertainty": self.get_uncertainty,
            "best_expected": self.calculate_best_expected_value,
            "best_largest": self.best_largest_value_observed,
            "max_uncertainty": self.max_path_uncertainty,
            "most_promising": self.on_most_promising_path,
            "second_most_promising": self.on_second_promising_path,
            "click_count": self.get_seq_click_count,
            "level_count": self.get_level_count,
            "branch_count": self.get_branch_count,
            "soft_pruning": self.soft_pruning,
            "first_observed": self.trial.first_node_observed,
            "count_observed_node_branch": self.count_observed_node_branch,
            "get_level_observed_std": self.get_level_observed_std,
            "successor_uncertainty": self.total_successor_uncertainty,
            "num_clicks_adaptive": self.trial.get_num_clicks,
            "num_clicks": self.trial.get_num_clicks,
            "max_expected_return": self.calculate_max_expected_return,
            "trial_level_std": self.get_trial_level_std,
            "soft_satisficing": self.soft_satisficing,
            "constant": self.constant_feature,
            "planning": self.constant_feature,
            "termination_constant": self.term_feature,
            "value": self.get_value,
            "is_observed": self.is_observed,
            "return_if_terminating": self.max_expected_if_terminal,
            "first_level": self.first_level,
            "second_level": self.second_level,
            "third_level": self.third_level,
            "avoid_first_level": self.avoid_first_level,
            "avoid_second_level": self.avoid_second_level,
            "avoid_third_level": self.avoid_third_level,
        })

        self.termination_map = OrderedDict({
            "is_max_path_observed": self.trial.termination_max_observed,  # is a feature
            "are_max_paths_observed": self.trial.termination_max_paths_observed,  # is a feature
            "is_previous_max": self.trial.termination_previous_max,  # is a feature
            "is_positive_observed": self.trial.termination_positive,  # is a feature
            "all_roots_observed": self.trial.termination_roots_observed,  # is a feature
            "all_leaf_nodes_observed": self.trial.termination_leaves_observed,  # is a feature
            "immediate_termination": self.trial.immediate_termination,  # NOT a feature
            "positive_root_leaves_termination": self.trial.termination_positive_root_leaves,
            "single_path_completion": self.trial.termination_single_path,  # is a feature
            "first_observed": self.trial.termination_first_node,  # NOT a feature
            "max_expected_return": self.calculate_max_expected_return,  # is a feature
            "soft_satisficing": self.trial.soft_satisficing,  # is a feature
            "constant": self.constant_feature,  # somehow there twice
            "termination_after_observing_positive_inner_and_one_outer": self.trial.termination_after_observing_positive_inner_and_one_outer
        })

    def first_level(self):
        if self.depth == 1:
            return 1
        return 0

    def second_level(self):
        if self.depth == 2:
            return 1
        return 0

    def third_level(self):
        if self.depth == 3:
            return 1
        return 0

    def avoid_first_level(self):
        if self.depth == 1:
            return 0
        return 1

    def avoid_second_level(self):
        if self.depth == 2:
            return 0
        return 1

    def avoid_third_level(self):
        if self.depth == 3:
            return 0
        return 1

    def is_traversed(self):
        return None  # todo: how to label as traversed?

    def observe(self):
        self.observed = True
        self.trial.set_previous_node(self)
        self.trial.observed_nodes.append(self)
        self.trial.unobserved_nodes.remove(self)
        if self.trial.sequence:
            self.trial.sequence.increment_count(self.label)
            if not self.root == self:
                self.trial.sequence.observed_node_values[
                    self.trial.node_level_map[self.label]
                ].append(self.value)

    def is_root(self):
        if self.parent is self.root:
            return 1
        return 0

    def is_leaf(self):
        if not self.children:
            return 1
        return 0

    def get_ancestor_nodes(self):
        if self.label == 0:
            return []
        ancestor_list = []
        node = self.parent
        while node.label != 0:
            ancestor_list.append(node)
            node = node.parent
        return ancestor_list

    def term_feature(self):
        if self.root == self:
            return 0
        else:
            return 1

    def is_observed(self):
        if self.observed:
            return 1
        else:
            return 0

    def get_value(self):
        if self.observed:
            return self.value
        else:
            return 0  # Change this later to expected value

    def get_ancestor_node_values(self):
        ancestor_list = self.get_ancestor_nodes()
        ancestor_values = [node.value for node in ancestor_list]
        return ancestor_values

    def get_observed_ancestor_nodes(self):
        ancestor_nodes = self.get_ancestor_nodes()
        observed_ancestor_nodes = []
        for node in ancestor_nodes:
            if node.observed:
                observed_ancestor_nodes.append(node)
        return observed_ancestor_nodes

    def get_observed_ancestor_node_values(self):
        observed_ancestor_nodes = self.get_observed_ancestor_nodes()
        return [node.value for node in observed_ancestor_nodes]

    def get_max_ancestor_value(self):
        ancestor_list = self.get_observed_ancestor_node_values()
        if not ancestor_list:
            return 0
        return max(ancestor_list)

    def get_observed_ancestor_count(self):
        return len(self.get_observed_ancestor_node_values())

    def is_leaf_and_positive_ancestor(self):
        if not self.is_leaf():
            return 0
        ancestor_nodes = self.get_observed_ancestor_nodes()
        for node in ancestor_nodes:
            if node.value > 0:
                return 1
        return 0

    def get_observed_node_count(self):
        observed_nodes = self.trial.get_observed_nodes()
        return len(observed_nodes)

    def get_unobserved_node_count(self):
        unobserved_nodes = self.trial.get_unobserved_nodes()
        return len(unobserved_nodes)

    def get_successor_nodes(self):
        def get_successors(node):
            node_list = []
            children = node.children
            if not children:
                return []
            for child in children:
                node_list.append(child)
                successors = get_successors(child)
                node_list += successors
            return node_list

        return get_successors(self)

    def get_successor_node_values(self):
        successor_list = self.get_successor_nodes()
        successor_values = [node.value for node in successor_list]
        return successor_values

    def get_observed_successor_nodes(self):
        successor_nodes = self.get_successor_nodes()
        observed_successor_nodes = []
        for node in successor_nodes:
            if node.observed:
                observed_successor_nodes.append(node)
        return observed_successor_nodes

    def get_observed_successor_node_values(self):
        observed_successor_nodes = self.get_observed_successor_nodes()
        return [node.value for node in observed_successor_nodes]

    def get_max_successor_value(self):
        successor_list = self.get_observed_successor_node_values()
        if not successor_list:
            return 0
        return max(successor_list)

    def get_observed_successor_count(self):
        return len(self.get_observed_successor_node_values())

    def sq_successor_count(self):
        return self.get_observed_successor_count() ** 2

    def get_immediate_successors(self):
        children = self.children
        children_list = []
        for child in children:
            if child.observed:
                children_list.append(child)
        return children_list

    def get_immediate_successor_count(self):
        immediate_successors = self.get_immediate_successors()
        return len(immediate_successors)

    def get_max_immediate_successor(self):
        immediate_successors = self.get_immediate_successors()
        if len(immediate_successors) == 0:
            return 0
        node_value_list = [node.value for node in immediate_successors]
        return max(node_value_list)

    def get_sibling_nodes(self):
        if not self.parent:
            return []
        if self.parent:
            node_list = self.parent.children.copy()
            node_list.remove(self)
            return node_list

    def get_observed_siblings(self):
        sibling_nodes = self.get_sibling_nodes()
        if not sibling_nodes:
            return []
        else:
            return [node for node in sibling_nodes if node.observed]

    def get_observed_siblings_count(self):
        return len(self.get_observed_siblings())

    def get_observed_same_depth_nodes(self):
        nodes_at_depth = self.trial.level_map[self.trial.node_level_map[self.label]]
        node_list = []
        for node in nodes_at_depth:
            if node.observed:
                node_list.append(node)
        return node_list

    def get_observed_same_depth_count(self):
        return len(self.get_observed_same_depth_nodes())

    def is_parent_observed(self):
        if self.parent.observed:
            return 1
        else:
            return 0

    def is_previous_successor(self):
        if self.trial.previous_observed:
            if self.trial.previous_observed in self.get_successor_nodes():
                return 1
        return 0

    def is_previous_observed_successor_negative(self):
        previous_node = self.trial.previous_observed
        if previous_node:
            successors = self.get_successor_nodes()
            if previous_node.value < 0 and previous_node in successors:
                return 1
        return 0

    def get_trial_level_std(self):
        trial = self.trial
        observed_level_values = trial.sequence.observed_node_values[
            trial.node_level_map[self.label]
        ]
        if len(observed_level_values) != 0:
            return np.std(observed_level_values)
        else:
            return 0

    def get_seq_click_count(self):
        return self.trial.sequence.get_click_count(self.label)

    def get_level_count(self):
        trial = self.trial
        nodes_at_same_level = trial.level_map[trial.node_level_map[self.label]]
        level_count = sum([node.get_seq_click_count() for node in nodes_at_same_level])
        return level_count

    def get_branch_count(self):
        trial = self.trial
        branches = trial.reverse_branch_map[self.label]
        branch_counts = []
        for branch in branches:
            node_list = trial.branch_map[branch]
            branch_sum = sum(
                [trial.node_map[node].get_seq_click_count() for node in node_list]
            )
            branch_counts.append(branch_sum)
        return max(branch_counts)

    def count_observed_node_branch(self):
        """What is the minimum of the number of observed nodes
        on branches that pass through the given node"""
        trial = self.trial
        branches = trial.reverse_branch_map[self.label]
        branch_counts = []
        for branch in branches:
            count = 0
            node_list = trial.branch_map[branch]
            for node in node_list:
                if trial.node_map[node].observed:
                    count += 1
            branch_counts.append(count)
        return min(branch_counts)

    def constant_feature(self):
        # Also referred to as "offset"
        return 1

    def soft_pruning(self):
        branches = self.trial.reverse_branch_map[self.label]
        branch_losses = []
        for branch in branches:
            node_list = self.trial.branch_map[branch]
            vals = []
            for node_num in node_list:
                node = self.trial.node_map[node_num]
                if node.observed:
                    vals.append(node.value)
            if not len(vals) == 0:
                branch_losses.append(min(vals))
        if not len(branch_losses) == 0:
            max_loss = min(branch_losses)
        else:
            max_loss = 0
        return max_loss

    def hard_pruning(self, threshold):
        max_loss = self.soft_pruning()
        if max_loss <= threshold:
            return -1
        else:
            return 0

    def get_level_observed_std(self):
        node_list = self.get_observed_same_depth_nodes()
        if len(node_list) == 0:
            return 0
        else:
            return np.std([node.value for node in node_list])

    def calculate_best_expected_value(self):
        expected_path_values = self.trial.get_path_expected_values()
        node_paths = self.trial.reverse_branch_map[self.label]
        return max([expected_path_values[path] for path in node_paths])

    def calculate_max_expected_return(self):
        expected_path_values = self.trial.get_path_expected_values()
        return max(expected_path_values.values())

    def best_largest_value_observed(self):
        largest_values = self.trial.largest_value_observed()
        node_paths = self.trial.reverse_branch_map[self.label]
        return max([largest_values[path] for path in node_paths])

    def get_depth_node(self):
        return self.trial.node_level_map[self.label]

    def hard_satisficing(self, aspiration_level):
        return 0

    def soft_satisficing(self):
        expected_path_values = self.trial.get_path_expected_values()
        max_expected_value = -max(list(expected_path_values.values()))
        return max_expected_value

    def get_parent_value(self):
        if self.parent.observed:
            return self.parent.value
        else:
            if self.parent == self.root:
                return 0
            else:
                return self.parent.expected_value

    def is_successor_highest_leaf(self):
        if not self.children:
            return 0
        max_value = self.trial.get_max_dist_value()
        successor_list = self.get_observed_successor_nodes()
        for successor in successor_list:
            if successor.value >= max_value:
                return 1
        return 0

    def on_most_promising_path(self):
        expected_path_values = self.trial.get_path_expected_values()
        node_paths = self.trial.reverse_branch_map[self.label]
        best_paths = [
            k
            for k, v in expected_path_values.items()
            if v == max(expected_path_values.values())
        ]
        for node_path in node_paths:
            if node_path in best_paths:
                return 1
        return 0

    def on_second_promising_path(self):
        expected_path_values = self.trial.get_path_expected_values()
        node_paths = self.trial.reverse_branch_map[self.label]
        sorted_values = sorted(set(expected_path_values.values()), reverse=True)
        for node_path in node_paths:
            if len(sorted_values) > 1:
                if expected_path_values[node_path] == sorted_values[1]:
                    return 1
        return 0

    def get_uncertainty(self):
        return self.trial.uncertainty_by_depth[self.depth]

    def total_successor_uncertainty(self):
        successor_nodes = self.get_successor_nodes()
        stds = []
        for s_node in successor_nodes:
            if s_node.observed:
                stds.append(0)
            else:
                stds.append(s_node.get_uncertainty())
        if stds:
            return sum(stds)
        else:
            return 0

    def max_path_uncertainty(self):
        """Gives the maximum value of uncertainities of paths that
        pass through the given node"""
        trial = self.trial
        node_paths = trial.reverse_branch_map[self.label]
        uncertainties = []
        for path_num in node_paths:
            total_uncertainty = 0
            for node in trial.branch_map[path_num]:
                if node == 0:
                    continue
                full_node = trial.node_map[node]
                if full_node.observed:
                    node_variance = 0
                else:
                    node_variance = trial.variance_by_depth[full_node.depth]
                total_uncertainty += node_variance
            uncertainties.append(total_uncertainty)
        return np.sqrt(max(uncertainties))

    def get_observed_height(self):
        if not self.children:
            return 0
        node_list = []
        for node in self.children:
            if node.observed:
                node_list.append(1 + node.get_observed_height())
        if not node_list:
            return 0
        return max(node_list)

    def calculate_max_improvement(self):
        other_paths, best_paths = self.trial.get_improvement_expected_values()
        best = 0
        worst = 0
        if len(other_paths) > 0:
            best = max(other_paths)
        if len(best_paths) > 0:
            worst = max(best_paths)
        if len(best_paths) == 0 or len(other_paths) == 0:
            return 0
        return best - worst

    def calculate_expected_improvement(self):
        other_paths, best_paths = self.trial.get_improvement_expected_values()
        best = 0
        worst = 0
        if len(other_paths) > 0:
            best = mean(other_paths)
        if len(best_paths) > 0:
            worst = mean(best_paths)
        if len(best_paths) == 0 or len(other_paths) == 0:
            return 0
        return best - worst

    def calculate_max_expected_return_information(self, level_values):
        expected_path_values = self.trial.get_path_expected_values_information(
            level_values
        )
        return max(expected_path_values.values())

    def max_expected_if_terminal(self):
        return self.calculate_max_expected_return() * (1 - self.term_feature())

    def is_path_to_leaf_observed(self):
        if not self.children:
            return 1
        results = []
        for child in self.children:
            results.append(child.is_path_to_leaf_observed())
        count = sum(results)
        return 1 if count > 0 else 0

    def check_path_observed(self):
        if not self.observed:
            return 0
        present_node = self
        while present_node.parent is not None:
            if not present_node.parent.observed:
                return 0
            present_node = present_node.parent
        if not self.is_path_to_leaf_observed():
            return 0
        return 1

    def list_all_features(self):
        lis = list(self.feature_function_map.keys())
        lis.remove("first_observed")
        lis.remove("immediate_termination")
        max_value = self.trial.get_max_dist_value()
        for pruning_threshold in [-48, -24, -8, 0]:
            lis.append(f"hp_{pruning_threshold}")
        for satisficing_threshold in np.linspace(0, max_value, num=7):
            lis.append(f"hs_{satisficing_threshold}")
        sorted_features_list = sorted(lis)
        return sorted_features_list

    def compute_feature_list_values(self, features, adaptive_satisficing={}):
        """
        NOTE: unused in this package.
        """
        evaluated_features = [1]
        for feature in features:
            if (
                    feature[:2] != "hp"
                    and feature[:2] != "hs"
                    and feature != "num_clicks_adaptive"
            ):
                evaluated_features.append(self.feature_function_map[feature]())
            elif feature[:2] == "hp":
                evaluated_features.append(self.hard_pruning(float(feature[3:])))
            elif feature[:2] == "hs":
                evaluated_features.append(self.hard_satisficing(float(feature[3:])))
            elif feature == "num_clicks_adaptive":
                if self == self.root:
                    if len(adaptive_satisficing) != 0:
                        aspiration_level = sigmoid(
                            self.calculate_max_expected_return()
                            - adaptive_satisficing["a"]
                            + adaptive_satisficing["b"]
                            * self.feature_function_map[feature]()
                        )
                        evaluated_features.append(
                            self.hard_satisficing(aspiration_level)
                        )
                    else:
                        evaluated_features.append(self.feature_function_map[feature]())
        return evaluated_features

    def compute_termination_feature_values(self, features, adaptive_satisficing={}):
        """
        Prefixes:
            - 'hp' = hard pruning
            - 'hs' = hard satisficing

        Args:
            features: features to consider, as list
            adaptive_satisficing: not really sure, if provided used in number of clicks (adaptive) feature #TODO

        Returns: feature values

        """
        evaluated_features = []
        features_regardless_terminal = [
            "max_expected_return",
            "constant",
            "return_if_terminating",
            "num_clicks",
        ]
        for feature in features:
            # if feature is neither hard-pruned or satisficed and not one of these "special ones"
            if (
                    feature[:2] != "hp"
                    and feature[:2] != "hs"
                    and feature != "num_clicks_adaptive"
                    and feature != "soft_satisficing"
                    and feature not in features_regardless_terminal
            ):
                # if we're not in a terminal state
                if not (self == self.root):
                    # if the feature is only for terminal states, put the value to -1
                    if feature in self.termination_map.keys():
                        # TODO why isn't this 0?
                        evaluated_features.append(-1)
                    # otherwise, evaluate from feature map
                    else:
                        evaluated_features.append(self.feature_function_map[feature]())
                # if we're in a terminal state
                else:
                    # evaluate from termination map
                    if feature in self.termination_map.keys():
                        evaluated_features.append(self.termination_map[feature]())
                    else:
                        evaluated_features.append(0)
            # if hard pruned, pass through that function (same terminal logic)
            elif feature[:2] == "hp":
                if not (self == self.root):
                    evaluated_features.append(self.hard_pruning(float(feature[3:])))
                else:
                    evaluated_features.append(0)
            # if hard satisficing, pass through that function (same terminal logic)
            elif feature[:2] == "hs":
                if self == self.root:
                    evaluated_features.append(
                        self.trial.hard_satisficing(float(feature[3:]))
                    )
                else:
                    evaluated_features.append(-1)
            elif feature == "soft_satisficing":
                # TODO it seems this feature could just be in the first if (?)
                if not self == self.root:
                    evaluated_features.append(self.feature_function_map[feature]())
                else:
                    evaluated_features.append(0)
            elif feature == "num_clicks_adaptive":
                # TODO it seems like the num_clicks_adaptive function could take the adaptive_satisficing argument
                # so it wouldn't look like a num_clicks feature to people trying to modify the code
                if not (self == self.root):
                    if len(adaptive_satisficing) != 0:
                        as_value = sigmoid(
                            self.calculate_max_expected_return()
                            - adaptive_satisficing["a"]
                            + (
                                    adaptive_satisficing["b"]
                                    * self.feature_function_map[feature]()
                            )
                        )
                        evaluated_features.append(-1 * as_value)
                    else:
                        evaluated_features.append(0)
                else:
                    evaluated_features.append(0)
            elif feature in features_regardless_terminal:
                evaluated_features.append(self.feature_function_map[feature]())
        return evaluated_features
