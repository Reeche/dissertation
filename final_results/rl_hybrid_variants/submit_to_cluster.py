from condor_utils import submit_sub_file

bid = 25


# reinforce variants without hierarchical models
models = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490]
conditions = ["strategy_discovery"]

with open("parameters.txt", "w") as parameters:
    for condition in conditions:
        for model in models:
            args = [condition, model]
            args_str = " ".join(str(x) for x in args) + "\n"
            parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
