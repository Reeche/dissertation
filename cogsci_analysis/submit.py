from condor_utils import create_sub_file, submit_sub_file

bid = 1  # The bid that you want to place
# script = "create_df.py"  # The file that you want to run on the cluster.

# exp_num_list = ["v1.0",
#                 "c2.1",
#                 "c1.1",
#                 "high_variance_high_cost",
#                 "high_variance_low_cost",
#                 "low_variance_high_cost",
#                 "low_variance_low_cost"]
exp_num_list = ["c1.1"]

# model batches
# model_batch_1 = list(range(0, 250))
# model_batch_2 = list(range(250, 500))
# model_batch_3 = list(range(500, 750))
# model_batch_4 = list(range(750, 1000))
# model_batch_5 = list(range(1000, 1250))
# model_batch_6 = list(range(1250, 1500))
# model_batch_7 = list(range(1500, 1750))
# model_batch_8 = list(range(1750, 2016))
model_batch_dict = {"model_batch_1": list(range(0, 5)),
                    "model_batch_2": list(range(5, 10))}
                    # "model_batch_3": list(range(500, 750)),
                    # "model_batch_4": list(range(750, 1000)),
                    # "model_batch_5": list(range(1000, 1250)),
                    # "model_batch_6": list(range(1250, 1500)),
                    # "model_batch_7": list(range(1500, 1750)),
                    # "model_batch_8": list(range(1750, 2016))}


with open("parameters.txt", "w") as parameters:
    for exp_num_ in exp_num_list:
        for key, values in model_batch_dict.items():
            args = [exp_num_, values, key]
            args_str = " ".join(str(x) for x in args) + "\n"
            parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
