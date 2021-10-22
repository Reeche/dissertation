from condor_utils import create_sub_file, submit_sub_file

num_p = 200  # Number of participants in the experiment (max pid + 1)
exps = ["v1.0"]  # Experiment number according to the data directory
for exp in exps:
    args = [exp]
    sub_file = create_sub_file(
        "exp_analysis.py", args, process_arg=True, num_runs=num_p, num_cpus=1
    )
    submit_sub_file(sub_file, 20)
