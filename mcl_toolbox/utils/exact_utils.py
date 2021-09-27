from contexttimer import Timer

from mcl_toolbox.utils.exact import solve
from mcl_toolbox.utils.env_utils import (
    get_all_possible_sa_pairs_for_env,
    get_sa_pairs_from_states,
    get_all_possible_states_for_ground_truths,
)


def timed_solve_env(env, verbose=True, save_q=False, ground_truths=None):
    """
    Solves environment, saves elapsed time and optionally prints value and elapsed time
    :param env: MouselabEnv with only discrete distribution (must not be too big)
    :param verbose: Whether or not to print out solve information once done
    :return: Q, V, pi, info
             Q, V, pi are all recursive functions
             info contains the number of times Q and V were called
                as well as the elapsed time ("time")
    """
    with Timer() as t:
        Q, V, pi, info = solve(env)
        info["time"] = t.elapsed
        if verbose:
            print("optimal -> {:.2f} in {:.3f} sec".format(V(env.init), t.elapsed))
        elif save_q:
            V(env.init)  # call V to cache q_dictionary

        #  Save Q function
        if save_q is not None and ground_truths is not None:
            # In some cases, it is too costly to save whole Q function
            info["q_dictionary"] = construct_partial_q_dictionary(Q, env, ground_truths)
        elif save_q is not None:
            info["q_dictionary"] = construct_q_dictionary(Q, env, verbose)

    return Q, V, pi, info


def construct_q_dictionary(Q, env, verbose=False):
    """
    Construct q dictionary for env, given environment is solved
    """

    sa = get_all_possible_sa_pairs_for_env(env, verbose=verbose)
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary


def construct_partial_q_dictionary(Q, env, selected_ground_truths):
    """
    Construct q dictionary for only specified ground truth values
    """
    all_possible_states = get_all_possible_states_for_ground_truths(
        env, selected_ground_truths
    )
    sa = get_sa_pairs_from_states(all_possible_states)
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary