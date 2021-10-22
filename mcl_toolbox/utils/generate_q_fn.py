from pathlib import Path

from mcl_toolbox.env.mouselab import MouselabEnv
from mcl_toolbox.utils.distributions import Categorical
from mcl_toolbox.utils.exact_utils import timed_solve_env
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.learning_utils import pickle_save


def reward(depth):
    if depth == 1:
        return Categorical([-4, -2, 2, 4])
    elif depth == 2:
        return Categorical([-8, -4, 4, 8])
    elif depth == 3:
        return Categorical([-48, -24, 24, 48])


if __name__ == "__main__":
    cost = -1
    BRANCHING = [3, 1, 2]
    env = MouselabEnv.new_symmetric(BRANCHING, reward, seed=0, cost=cost)
    exp_num = "v1.0"
    E = Experiment(exp_num)
    ground_truths = list(
        set([tuple(e) for p in E.participants.values() for e in p.envs])
    )
    ground_truths = [list(gt) for gt in ground_truths]
    Q, V, pi, info = timed_solve_env(
        env, save_q=True, verbose=True, ground_truths=ground_truths
    )
    file_location = Path(__file__).parents[1]
    path = file_location.joinpath(f"data/{exp_num}_q.pkl")
    pickle_save(info, path)
