from condor_utils import submit_sub_file

bid = 25

exp_num = ['strategy_discovery']
# recovered_model_type = ['habitual', 'mf', 'non_learning']
# models = [491, 3326, 1743, 1756]

## variants
# recovered_model_type = ["variants/3326", "variants/3325", "variants/3324", "variants/3323",
#                         "variants/3318", "variants/3317", "variants/3316", "variants/3315"]

# which models to recover for
recovered_model_type = ["variants/3326", "variants/3325", "variants/3324", "variants/3318", "variants/3316"]
# which models should be fitted
models = [3326, 3325, 3324, 3323, 3318, 3317, 3316, 3315]

test_pid_dict = {
    'v1.0': [5],
    'c2.1': [0],
    'c1.1': [2],
    'high_variance_high_cost': [83],
    'high_variance_low_cost': [53],
    'low_variance_high_cost': [61],
    'low_variance_low_cost': [42],
    'strategy_discovery': [3]
}

hybrid_reinforce_pid_dict = {
    'v1.0': [5, 43, 82, 137, 154],
    'c2.1': [0, 13, 78, 166],
    'c1.1': [2, 14, 125, 171],
    'high_variance_high_cost': [83, 195],
    'high_variance_low_cost': [53],
    'low_variance_high_cost': [61, 79, 100, 107, 128, 132, 166, 206],
    'low_variance_low_cost': [42, 110, 172],
    'strategy_discovery': [3, 4, 6, 7, 9, 16, 17, 19, 23, 30, 34, 35, 41, 45, 53, 57, 58, 67, 71, 76,
                           78, 83, 86, 92, 106, 128, 133, 138, 139, 141, 143, 146, 155, 161, 164, 165,
                           167, 174, 175, 177, 184, 189, 194, 195, 201, 203, 206, 211, 216, 218, 219, 223,
                           228, 231, 232, 236, 238, 250, 255, 259, 260, 262, 267, 280, 281, 291, 292, 293,
                           299, 305, 310, 316, 317, 318, 320, 324, 327, 328, 341, 344, 347, 349, 350, 355,
                           356, 357, 359, 360, 361, 362, 373, 374, 375, 377]  # n=94
}

mf_reinforce_pid_dict = {
    'v1.0': [15, 104, 148, 150, 158],
    'c2.1': [25, 31, 64, 96, 123, 128, 133, 136, 142],
    'c1.1': [7, 9, 27, 28, 42, 44, 48, 139, 163],
    'high_variance_high_cost': [108],
    'high_variance_low_cost': [197],
    'low_variance_high_cost': [2, 14, 36, 73, 98, 135, 138, 144, 157, 171, 181, 183],
    'low_variance_low_cost': [115, 137, 143, 165, 170],
    'strategy_discovery': [2, 8, 24, 43, 48, 49, 54, 62, 68, 73, 75, 80, 85, 91, 93, 96, 99, 102, 107, 110, 113, 116,
                           117, 120, 123, 124, 126, 131, 137, 145, 147, 149, 153, 156, 159, 166, 169, 171, 172, 178,
                           181, 183, 185, 187, 190, 199, 200, 207, 212, 213, 220, 221, 226, 229, 233, 242, 244, 246,
                           247, 252, 261, 263, 266, 274, 279, 286, 287, 294, 295, 296, 306, 319, 333, 337, 340, 365,
                           367, 369, 372, 376, 378]  # n=81
}
habitual_pid_dict = {
    'v1.0': [1, 17, 29, 34, 38, 45, 62, 66, 80, 85, 90, 110, 155],
    'c2.1': [26, 84, 99, 113, 145, 152, 162],
    'c1.1': [12, 23, 83, 111, 116, 147],
    'high_variance_high_cost': [1, 76, 169, 191],
    'high_variance_low_cost': [35, 96],
    'low_variance_high_cost': [28, 37, 45, 69, 147],
    'low_variance_low_cost': [85, 91, 106, 186],
    'strategy_discovery': [1, 10, 11, 14, 20, 22, 25, 26, 27, 29, 33, 36, 37, 39, 40, 46, 50, 51, 52, 55, 59, 65, 70,
                           89, 95, 98, 101, 111, 115, 118, 119, 125, 129, 134, 135, 140, 142, 148, 151, 154, 162, 170,
                           180, 186, 192, 193, 202, 204, 205, 209, 210, 214, 215, 217, 234, 235, 237, 240, 241, 249,
                           253, 254, 257, 265, 268, 271, 276, 277, 282, 289, 300, 304, 308, 312, 313, 321, 322, 323,
                           329, 330, 331, 332, 339, 343, 348, 358, 363, 364, 370]  # n=89
}
non_learning_pid_dict = {
    'v1.0': [6, 10, 18, 24, 56, 68, 69, 94, 106, 144, 146, 165, 173],
    'c2.1': [8, 16, 20, 22, 30, 39, 41, 49, 52, 53, 58, 60, 61, 67, 72, 86, 88, 93,
             95, 107, 108, 115, 122, 130, 134, 138, 149, 156, 164, 170, 172],
    'c1.1': [19, 36, 37, 50, 54, 65, 70, 71, 74, 81, 89, 92, 100, 102, 105, 109, 114, 131,
             135, 143, 151, 159, 167, 168],
    'high_variance_high_cost': [0, 32, 47, 57, 74, 81],
    'high_variance_low_cost': [17, 23, 154, 180],
    'low_variance_high_cost': [21, 31, 124, 201],
    'low_variance_low_cost': [12, 19, 27, 44, 52, 77, 104, 113, 130, 179, 184, 196, 200],
    'strategy_discovery': [18, 28, 32, 38, 56, 63, 72, 77, 82, 90, 103, 109, 122, 152, 173, 196, 239, 256, 275, 278,
                           309, 311, 315, 335, 336, 342, 346, 352, 353, 354, 371]  # n=31
}

pid_dict = {
    'v1.0': [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 51, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77,
             80, 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 121, 124, 126, 132, 137, 140, 141, 144, 146,
             148, 150, 154, 155, 158, 160, 165, 169, 173],
    'c2.1': [0, 3, 8, 11, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78,
             79, 84, 86, 88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138,
             142, 145, 149, 152, 156, 162, 164, 166, 170, 172],
    'c1.1': [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 46, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81,
             83, 87, 89, 91, 92, 97, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147,
             151, 153, 157, 159, 161, 163, 167, 168, 171],
    'high_variance_high_cost': [0, 1, 10, 18, 22, 25, 30, 32, 38, 41, 46, 47, 49, 57, 60, 63, 65, 70, 74, 76, 81, 83,
                                88, 89, 94, 103, 108, 109, 111, 114, 116, 118, 125, 129, 134, 139, 149, 150, 156, 161,
                                164, 167, 169, 173, 177, 182, 188, 191, 195, 198, 199, 204],
    'high_variance_low_cost': [4, 7, 8, 17, 20, 23, 26, 29, 33, 35, 40, 48, 50, 51, 53, 56, 58, 64, 71, 78, 82, 87, 92,
                               93, 95, 96, 101, 112, 117, 119, 122, 126, 131, 133, 136, 141, 145, 146, 151, 154, 158,
                               162, 168, 175, 180, 185, 187, 189, 193, 197, 202, 205],
    'low_variance_high_cost': [2, 13, 14, 16, 21, 24, 28, 31, 36, 37, 43, 45, 54, 61, 62, 68, 69, 73, 79, 80, 84, 86,
                               90, 97, 98, 100, 102, 107, 120, 124, 128, 132, 135, 138, 140, 144, 147, 153, 157, 160,
                               163, 166, 171, 174, 181, 183, 192, 194, 201, 203, 206],
    'low_variance_low_cost': [3, 5, 6, 9, 11, 12, 15, 19, 27, 34, 39, 42, 44, 52, 55, 59, 66, 67, 72, 75, 77, 85, 91,
                              99, 104, 105, 106, 110, 113, 115, 121, 123, 127, 130, 137, 142, 143, 148, 152, 155, 159,
                              165, 170, 172, 176, 178, 179, 184, 186, 190, 196, 200, 207],
    'strategy_discovery': list(range(1, 379))}

## for SD, only fit the necessary participants for the variants
sd_variants_dict = {
    # these are also the particpants who are best explained by hybrid Reinforce
    '3326': [3, 4, 6, 7, 9, 16, 17, 19, 23, 30, 34, 35, 67, 71, 83, 92, 106, 128, 138, 139, 141, 143, 146, 155, 161,
             165, 167, 174, 177, 195, 201, 206, 211, 218, 223, 228, 232, 236, 250, 260, 262, 267, 280, 281, 291, 292,
             299, 310, 316, 318, 324, 328, 341, 344, 347, 349, 350, 357, 359, 360, 361, 362, 373, 374, 375, 377],
    # n=66
    # SE participants who are best explained by hybrid reinforce as base mechanism: [53, 57, 76, 184, 189, 255, 327, 356]
    '3324': [2, 5, 11, 15, 18, 27, 28, 48, 50, 53, 55, 56, 57, 62, 69, 70, 72, 76, 81, 82, 88, 89, 103, 105, 109, 111,
             112, 115, 119, 122, 129, 130, 134, 137, 144, 151, 166, 169, 172, 180, 182, 184, 189, 196, 209, 220, 235,
             239, 241, 246, 249, 252, 255, 275, 283, 284, 285, 290, 311, 321, 326, 327, 332, 335, 340, 352, 354, 356,
             358, 378],  # n=70
    # PR participants who are best explained by hybrid reinforce as base mechanism: [78, 164, 175, 203, 216, 219, 231, 259, 293, 317, 320, 355]
    '3318': [8, 13, 20, 32, 38, 39, 40, 42, 61, 63, 65, 75, 78, 87, 91, 104, 125, 126, 140, 142, 148, 152, 154, 163,
             164, 168, 170, 171, 175, 181, 186, 192, 193, 200, 202, 203, 213, 214, 216, 219, 225, 231, 254, 257, 259,
             261, 268, 276, 282, 293, 294, 302, 312, 314, 317, 320, 330, 331, 336, 342, 355],  # n=61
    # PR+SE participants who are best explained by hybrid reinforce as base mechanism: [58, 86, 238]
    '3316': [12, 14, 22, 25, 36, 58, 66, 86, 90, 95, 101, 117, 173, 176, 197, 210, 224, 234, 237, 238, 240, 251, 256,
             258, 265, 271, 273, 274, 277, 278, 279, 289, 309, 313, 315, 323, 325, 339, 343, 346, 351, 353, 363],
    # n=43
    # TD participants who are best explained by hybrid reinforce as base mechanism: [41, 45, 133, 194, 305]
    '3325': [24, 37, 41, 43, 44, 45, 47, 54, 68, 80, 85, 93, 96, 98, 99, 102, 107, 113, 120, 123, 124, 131, 133, 145,
             149, 153, 156, 159, 162, 178, 185, 187, 190, 194, 198, 199, 207, 217, 221, 226, 229, 242, 244, 247, 253,
             263, 270, 288, 295, 296, 305, 306, 319, 333, 337, 365, 367, 372, 376]  # n=59
}



# with open("parameters.txt", "w") as parameters:
#     for recovered_model in recovered_model_type:
#         # if recovered_model == 'hybrid':
#         #     pid_dict = hybrid_reinforce_pid_dict
#         # elif recovered_model == 'mf':
#         #     pid_dict = mf_reinforce_pid_dict
#         # elif recovered_model == 'habitual':
#         #     pid_dict = habitual_pid_dict
#         # elif recovered_model == 'non_learning':
#         #     pid_dict = non_learning_pid_dict
#
#         recovered_model_formatted = recovered_model.replace("/", "")  # Replace / with _
#
#         for exp_num_ in exp_num:
#             if exp_num_ == 'strategy_discovery':
#                 num_trial = 120
#             else:
#                 num_trial = 35
#             for models_ in models:
#                 pids = pid_dict.get(exp_num_)
#                 for pid in pids:
#                     args = [exp_num_, models_, 'likelihood', pid, num_trial, recovered_model,
#                             recovered_model_formatted]
#                     args_str = " ".join(str(x) for x in args) + "\n"
#                     parameters.write(args_str)

### for SD variants
with open("parameters.txt", "w") as parameters:
    for recovered_model in recovered_model_type:
        recovered_model_formatted = recovered_model.replace("/", "")  # Replace / with _
        for exp_num_ in exp_num:
            if exp_num_ == 'strategy_discovery':
                num_trial = 120
            else:
                num_trial = 35
            for models_ in models:
                pids = sd_variants_dict.get(recovered_model.split("/")[1])
                for pid in pids:
                    args = [exp_num_, models_, 'likelihood', pid, num_trial, recovered_model,
                            recovered_model_formatted]
                    args_str = " ".join(str(x) for x in args) + "\n"
                    parameters.write(args_str)

submit_sub_file("sub_multiple.sub", bid)
