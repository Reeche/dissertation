from condor_utils import create_sub_file, submit_sub_file

model_nums = [
    0,
    1,
    64,
    65,
    128,
    129,
    576,
    577,
    640,
    641,
    704,
    705,
    1824,
    1825,
    1728,
    1729,
    1920,
    1921,
]

num_runs = 40
for model in model_nums:
    args = ["mer_performance_error", model]
    sub_file = create_sub_file(
        "fit_mcrl_models.py", args, process_arg=True, num_runs=num_runs, num_cpus=1
    )
    submit_sub_file(sub_file, 20)
