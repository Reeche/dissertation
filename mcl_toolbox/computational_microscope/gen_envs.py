import sys
from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.utils.learning_utils import pickle_load, pickle_save, create_dir


def gen_envs(pipeline):
    num_simulations = len(pipeline)
    env = GenericMouselabEnv(num_simulations, pipeline)
    E = env.ground_truth
    return E

if __name__ == "__main__":
    exp_num = sys.argv[1]
    num_simulations = int(sys.argv[2])

    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    exp_reward_structures = {'v1.0': 'high_increasing', 'F1': 'high_increasing', 
                            'c1.1_old': 'low_constant', 'T1.1': 'large_increasing'}

    # Defaults for 312 increasing variance task
    reward_structure = "high_increasing"
    pipeline = [exp_pipelines["v1.0"][0]]*num_simulations
    
    if exp_num in exp_reward_structures:
        pipeline = [exp_pipelines[exp_num][0]]*num_simulations
    else:
        if exp_num == "c2.1_inc":
            pipeline = [exp_pipelines["c2.1_inc"][0]]*num_simulations
        else:
            pipeline = [exp_pipelines["c2.1_dec"][0]]*num_simulations

    # normalized_features = get_normalized_features(reward_structure)
    E = gen_envs(pipeline)
    result_dir = f"results/cluster_envs/{exp_num}"
    create_dir(result_dir)
    pickle_save(E, f"{result_dir}/random_envs_{num_simulations}.pkl")