from condor_utils import submit_sub_file

bid = 25

## for testing
# exp_num = ['v1.0']
# models = [256]
# pid_dict = {
#     'v1.0': [121]}

exp_num = "strategy_discovery"

pids = [18]
# pids = list(range(1, 379))
num_trial = 120

with open("parameters.txt", "w") as parameters:
    for pid in pids:
        args = [pid, exp_num, num_trial]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple_cm.sub", bid)
