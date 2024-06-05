# from mouselab.mouselab import MouselabEnv
from mouselab.envs.registry import register, registry
from mouselab.distributions import Categorical
from .generic_mouselab import GenericMouselabEnv
import numpy as np
from toolz import get


class ConditionalMouselabEnv(GenericMouselabEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # possible_ground_truths = [
        #     (0, -1, -5, -5, -5, 1, -5, 50, -50, -1, -5, -5, -5),
        #     (0, -1, -5, -5, -5, 1, -5, -50, 50, -1, -5, -5, -5),
        #     (0, -1, -5, -5, -5, -1, -5, -5, -5, 1, -5, -50, 50),
        #     (0, -1, -5, -5, -5, -1, -5, -5, -5, 1, -5, 50, -50),
        #     (0, 1, -5, 50, -50, -1, -5, -5, -5, -1, -5, -5, -5),
        #     (0, 1, -5, -50, 50, -1, -5, -5, -5, -1, -5, -5, -5),
        # ]

        # if tuple(self.ground_truth) not in possible_ground_truths:
        #     raise ValueError(
        #         "Ground truth does not fit in with hard coded possible_ground_truths"
        #     )

    def results(self, state, action):
        """Returns a list of possible results_2000_iterations of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        if action == self.term_action:
            yield (1, self.term_state, self.expected_term_reward(state))
        elif self.include_last_action:
            # assume you are not using the distance cost or a cost that depends on last action
            raise NotImplementedError
        else:
            # check if branch is discovered
            # check if early node that is 1 is revealed, or +50 or -50
            found_reward_branch = 0
            found_rewarding_states = [
                action_index
                for action_index in range(len(self._state))
                # if self._state[action_index] in [+1, -50, +50]
                if state[action_index] in [+1, -50, +50]
            ]
            if len(found_rewarding_states) > 0:
                # should only be in one cluster
                assert (
                    len(
                        np.unique(
                            [
                                self.mdp_graph.nodes[rewarding_state]["cluster"]
                                for rewarding_state in found_rewarding_states
                            ]
                        )
                    )
                    <= 1
                )
                found_reward_branch = np.unique(
                    [
                        self.mdp_graph.nodes[rewarding_state]["cluster"]
                        for rewarding_state in found_rewarding_states
                    ]
                )[0]

            # if reward branching not yet found or action is depth = 2, it's original problem
            if (found_reward_branch == 0) or (
                self.mdp_graph.nodes[action]["depth"] == 2
            ):
                for r, p in state[action]:
                    s1 = list(state)
                    s1[action] = r
                    yield (p, tuple(s1), self.cost(state, action))
            # else we're not on the rewarding branch
            elif self.mdp_graph.nodes[action]["cluster"] != found_reward_branch:
                if self.mdp_graph.nodes[action]["depth"] == 1:
                    s1 = list(state)
                    s1[action] = -1
                    yield (1, tuple(s1), self.cost(state, action))
                else:
                    s1 = list(state)
                    s1[action] = -5
                    yield (1, tuple(s1), self.cost(state, action))
            # or we are on the rewarding branch
            else:
                if self.mdp_graph.nodes[action]["depth"] == 1:
                    s1 = list(state)
                    s1[action] = 1
                    yield (1, tuple(s1), self.cost(state, action))
                elif self.mdp_graph.nodes[action]["depth"] == 3:
                    if -50 in self._state:
                        s1 = list(state)
                        s1[action] = 50
                        yield (1, tuple(s1), self.cost(state, action))
                    elif 50 in self._state:
                        s1 = list(state)
                        s1[action] = -50
                        yield (1, tuple(s1), self.cost(state, action))
                    else:
                        for r, p in Categorical([50, -50], [1 / 2, 1 / 2]):
                            s1 = list(state)
                            s1[action] = r
                            yield (p, tuple(s1), self.cost(state, action))
                else:
                    raise AssertionError("Did not expect to get here")

    @classmethod
    def new_symmetric_registered(cls, experiment_setting, seed=None, **kwargs):
        branching = registry(experiment_setting).branching
        reward = registry(experiment_setting).reward_function

        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        tree = []

        def expand(d):
            my_idx = len(init)
            init.append(reward(d))
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d + 1)
                children.append(child_idx)
            return my_idx

        expand(0)
        return cls(tree, init, seed=seed, **kwargs)


if __name__ == "__main__":
    register(
        name="conditional",
        branching=[3, 1, 2],
        reward_inputs="depth",
        reward_dictionary={
            1: Categorical([-1, 1], [2 / 3, 1 / 3]),
            2: Categorical([-5], [1]),
            3: Categorical([-5, +50, -50], [2 / 3, 1 / 6, 1 / 6]),
        },
    )

    # inherit, using hard coded values and symmetric environment
    env = ConditionalMouselabEnv.new_symmetric_registered("conditional")

    print(list(env.results(env._state, 1)))
    env.step(1)
    print(list(env.results(env._state, 3)))
