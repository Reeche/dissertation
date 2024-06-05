from condor_utils import submit_sub_file

bid = 25


# conditions = ["v1.0", "c1.1", "c2.1"]
conditions = ["c1.1"]
models = ["level_level"]
# conditions = ["strategy_discovery"]
# models = [480, 481, 483, 484, 485, 487, 488, 489, 490] # rl_hybrid_variants
# models = ["522", "491", "479", "1743", "1756",
#           "no_assumption_individual", "no_assumption_level",
#           "uniform_individual", "uniform_level",
#           "level_individual", "level_level"]


with open("parameters.txt", "w") as parameters:
    for condition in conditions:
        for model in models:
            args = [condition, model]
            args_str = " ".join(str(x) for x in args) + "\n"
            parameters.write(args_str)

submit_sub_file("sub_multiple_cm.sub", bid)
