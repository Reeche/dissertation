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

hybrid_reinforce_pid_dict = {
    'v1.0': [5, 43, 82, 137, 154],
    'c2.1': [0, 13, 78, 166],
    'c1.1': [2, 14, 125, 171],
    'high_variance_high_cost': [83, 195],
    'high_variance_low_cost': [53],
    'low_variance_high_cost': [61, 79, 100, 107, 128, 132, 166, 206],
    'low_variance_low_cost': [42, 110, 172]
}

# three cond clicking participants who are best explained by hybrid reinforce according to BIC
threecond_hybrid_reinforce_dict = {
    'v1.0': [5, 43, 82, 137, 154],
    'c2.1': [0, 13, 78, 166],
    'c1.1': [2, 14, 125, 171],
}

planningamount_hybrid_reinforce_dict = {
    'high_variance_high_cost': [83, 195],
    'high_variance_low_cost': [53],
    'low_variance_high_cost': [61, 79, 100, 107, 128, 132, 166, 206],
    'low_variance_low_cost': [42, 110, 172],
}

mf_reinforce_pid_dict = {
    'v1.0': [15, 104, 148, 150, 158],
    'c2.1': [25, 31, 64, 96, 123, 128, 133, 136, 142],
    'c1.1': [7, 9, 27, 28, 42, 44, 48, 139, 163],
    'high_variance_high_cost': [108],
    'high_variance_low_cost': [197],
    'low_variance_high_cost': [2, 14, 36, 73, 98, 135, 138, 144, 157, 171, 181, 183],
    'low_variance_low_cost': [115, 137, 143, 165, 170]
}
habitual_pid_dict = {
    'v1.0': [1, 17, 29, 34, 38, 45, 62, 66, 80, 85, 90, 110, 155],
    'c2.1': [26, 84, 99, 113, 145, 152, 162],
    'c1.1': [12, 23, 83, 111, 116, 147],
    'high_variance_high_cost': [1, 76, 169, 191],
    'high_variance_low_cost': [35, 96],
    'low_variance_high_cost': [28, 37, 45, 69, 147],
    'low_variance_low_cost': [85, 91, 106, 186]
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
    'low_variance_low_cost': [12, 19, 27, 44, 52, 77, 104, 113, 130, 179, 184, 196, 200]
}

def assign_model_names(row):
    if str(row['model_index']) == "3326":
        return 'hybrid Reinforce'
    elif str(row['model_index']) == "491":
        return 'MF - Reinforce'
    elif str(row['model_index']) == "1743":
        return 'Habitual'
    elif str(row['model_index']) == "1756":
        return 'Non-learning'
    else:
        raise ValueError("Model class combination not found")

rename_index = {
    3326: 'hybrid Reinforce',
    491: 'MF - Reinforce',
    1743: 'Habitual',
    1756: 'Non-learning'
}

rename_map = {
    'hybrid_reinforce': 'hybrid Reinforce',
    'mf_reinforce': 'MF - Reinforce',
    'habitual': 'Habitual',
    'non_learning': 'Non-learning'
}

def assign_pid_dict(recovered_model):
    if recovered_model == "hybrid_reinforce":
        return hybrid_reinforce_pid_dict
    elif recovered_model == "mf_reinforce":
        return mf_reinforce_pid_dict
    elif recovered_model == "habitual":
        return habitual_pid_dict
    elif recovered_model == "non_learning":
        return non_learning_pid_dict


def process_clicks(row):
    return [len(sublist) - 1 for sublist in row]


def process_data(data, model_col, pid_col, exp):
    if exp == "strategy_discovery":
        data[pid_col] = data[pid_col].apply(
            lambda x: ast.literal_eval(re.sub(r'(?<=\d|\-)\s+(?=\d|\-)', ', ', x.replace('\n', ' '))) if isinstance(x,
                                                                                                                    str) else x)
        data[model_col] = data[model_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    elif pid_col == "pid_rewards":
        data[pid_col] = data[pid_col].apply(lambda x: [int(i) for i in x.strip("[]").split()])
    else:
        data[pid_col] = data[pid_col].apply(lambda x: ast.literal_eval(x))
        data[model_col] = data[model_col].apply(lambda x: ast.literal_eval(x))
    return data
