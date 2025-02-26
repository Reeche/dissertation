clicking_pid = {
    "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 62, 66, 68, 69, 73, 75, 77, 80,
             82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 140, 144, 146, 148, 150, 154, 155,
             158, 160, 165, 169, 173],
    "c2.1": [0, 8, 13, 16, 20, 22, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 72, 78, 79, 84, 86,
             88, 93, 95, 96, 99, 103, 107, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149,
             152, 156, 162, 164, 166, 170, 172],
    "c1.1": [2, 4, 7, 9, 12, 14, 19, 23, 27, 28, 32, 36, 37, 42, 44, 48, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 83, 89,
             91, 92, 100, 102, 105, 109, 111, 114, 116, 120, 125, 127, 129, 131, 135, 139, 143, 147, 151, 153, 157, 159,
             161, 163, 167, 168, 171]
}


def assign_model_names(row):
    if row['class'] == 'hybrid' and str(row['model_index']) in ["491", "3326"]:
        return 'hybrid Reinforce'
    elif row['class'] == 'hybrid' and str(row['model_index']) == "479":
        return 'hybrid LVOC'
    elif row['class'] in ['pure', 'mf'] and str(row['model_index']) == "491":
        return 'MF - Reinforce'
    elif row['class'] in ['pure', 'mf'] and str(row['model_index']) == "479":
        return 'MF - LVOC'
    elif str(row['model_index']) == "1743":
        return 'Habitual'
    elif str(row['model_index']) == "1756":
        return 'Non-learning'
    elif str(row['model_index']) == "522":
        return 'SSL'
    elif row['model_index'] == "no_assumption_level":
        return 'MB - Uniform, grouped'
    elif row['model_index'] == "no_assumption_individual":
        return 'MB - Uniform, ind.'
    elif row['model_index'] == "uniform_individual":
        return 'MB - Equal, ind.'
    elif row['model_index'] == "uniform_level":
        return 'MB - Equal, grouped'
    elif row['model_index'] == "level_level":
        return 'MB - Level, grouped'
    elif row['model_index'] == "level_individual":
        return 'MB - Level, ind.'
    else:
        raise ValueError("Model class combination not found")


threecond_learners = [5, 35, 43, 82, 117, 137, 154, 1, 17, 29, 34, 38, 45, 62, 66, 80, 85, 90, 110, 155, 15, 104, 112,
                      119, 148, 150, 158, 21, 40, 55, 59, 73, 75, 77, 98, 101, 124, 132, 140, 160, 169, 0, 13, 78, 166,
                      25, 31, 64, 96, 123, 128, 133, 136, 142, 26, 84, 99, 113, 145, 152, 162, 33, 47, 79, 103, 118, 4,
                      32, 63, 76, 91, 120, 127, 129, 153, 157, 161, 2, 14, 125, 171, 7, 9, 27, 28, 42, 44, 48, 57, 139,
                      163, 12, 23, 83, 111, 116, 147]
threecond_nonlearners = [6, 10, 18, 24, 56, 68, 69, 94, 106, 144, 146, 165, 173, 8, 16, 20, 22, 30, 39, 41, 49, 52, 53,
                         58, 60, 61, 67, 72, 86, 88, 93, 95, 107, 108, 115, 122, 130, 134, 138, 149, 156, 164, 170, 172,
                         19, 36, 37, 50, 54, 65, 70, 71, 74, 81, 89, 92, 100, 102, 105, 109, 114, 131, 135, 143, 151,
                         159, 167, 168]
