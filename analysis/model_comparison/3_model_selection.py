import pandas as pd
import numpy as np


"""
Compare vanilla models based on the fit
"""



def compare_pseudo_likelihood(data):
    BIC = 2 * data["click_loss"] + data["number_of_parameters"] * np.log(35)
    return BIC

def compare_number_of_clicks_likelihood(data):
    BIC = 2 * data["mer_loss"] + data["number_of_parameters"] * np.log(35)
    return BIC

def sort_by_BIC(data):
    df = data.sort_values(by=["BIC"])
    average_bic = df.groupby('model')['BIC'].mean().reset_index()
    sorted_df = average_bic.sort_values(by='BIC', ascending=True)
    # the smaller BIC the better
    print(sorted_df)
    return None

if __name__ == "__main__":
    exp = "v1.0"

    if exp in ["v1.0", "c1.1", "c2.1"]:
        criterion = 'pseudo_likelihood'
    else:
        criterion = 'number_of_clicks_likelihood'

    data = pd.read_csv(f"data/{exp}_{criterion}.csv", index_col=0)

    # add BIC
    if exp in ["v1.0", "c1.1", "c2.1"]:
        data["BIC"] = compare_pseudo_likelihood(data)
    else:
        data["BIC"] = compare_number_of_clicks_likelihood(data)

    # filter for adpative participants
    learning_participants = {
        "v1.0": [1, 5, 6, 10, 15, 17, 18, 21, 24, 29, 34, 35, 38, 40, 43, 45, 55, 56, 59, 66, 68, 69, 73, 75, 77, 80,
                 82, 85, 90, 94, 98, 101, 104, 106, 110, 112, 117, 119, 124, 132, 137, 144, 146, 150, 154, 155, 158,
                 160, 165, 169, 173],
        "c2.1": [0, 8, 13, 16, 20, 25, 26, 30, 31, 33, 39, 41, 47, 49, 52, 53, 58, 60, 61, 64, 67, 84, 86, 93, 95, 96,
                 99, 103, 108, 113, 115, 118, 122, 123, 128, 130, 133, 134, 136, 138, 142, 145, 149, 156, 164, 166,
                 170],
        "c1.1": [2, 4, 7, 9, 12, 19, 23, 27, 28, 32, 37, 42, 44, 50, 54, 57, 63, 65, 70, 71, 74, 76, 81, 89, 91, 92,
                 100, 102, 105, 109, 116, 120, 125, 127, 129, 131, 135, 139, 143, 151, 153, 157, 159, 161, 163, 167,
                 168, 171],
        "high_variance_high_cost": [32, 49, 57, 60, 94, 108, 109, 111, 129, 134, 139, 149, 161, 164, 191, 195],
        "high_variance_low_cost": [7, 8, 17, 23, 35, 48, 50, 51, 53, 58, 71, 82, 92, 93, 96, 101, 117, 126, 136, 145,
                                   146, 154, 158, 180, 189, 197],
        "low_variance_high_cost": [2, 13, 14, 24, 28, 36, 45, 61, 62, 69, 73, 79, 80, 86, 100, 107, 124, 128, 135, 138,
                                   160, 166, 171, 174, 183, 201, 206],
        "low_variance_low_cost": [9, 42, 52, 85, 110, 115, 143, 165, 172]}

    data = data[data["pid"].isin(learning_participants[exp])]

    sort_by_BIC(data)
