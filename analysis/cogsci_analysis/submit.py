from condor_utils import submit_sub_file

bid = 1  # The bid that you want to place
# script = "create_df.py"  # The file that you want to run on the cluster.

# exp_num_list = ["v1.0",
#                 "c2.1",
#                 "c1.1",
#                 "high_variance_high_cost",
#                 "high_variance_low_cost",
#                 "low_variance_high_cost",
#                 "low_variance_low_cost"]
exp_num_list = ["c2.1"]



with open("parameters.txt", "w") as parameters:
    for exp_num_ in exp_num_list:
        args = [exp_num_]
        args_str = " ".join(str(x) for x in args) + "\n"
        parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
