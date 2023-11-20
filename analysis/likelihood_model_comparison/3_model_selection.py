import pandas as pd
import numpy as np
from vars import learning_participants
# import pymc3 as pm

"""
Compare vanilla models based on the fit
"""


def compare_pseudo_likelihood(data, trials):
    BIC = 2 * data["click_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC


def compare_number_of_clicks_likelihood(data, trials):
    BIC = 2 * data["mer_loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC

def compare_loss(data, trials):
    BIC = 2 * data["loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC

def sort_by_BIC(data):
    df = data.sort_values(by=["BIC"])
    average_bic = df.groupby('model')['BIC'].mean().reset_index()
    sorted_df = average_bic.sort_values(by='BIC', ascending=True)
    # the smaller BIC the better
    print(sorted_df)
    return sorted_df


def bms(model_bic):
    # Step 1: Calculate BIC for each model
    # Assume bic_values is a list of pre-calculated BIC values for different models

    # Step 2: Compute the BIC differences
    delta_bic = np.array(model_bic["BIC"]) - min(model_bic["BIC"])

    # Step 3: Calculate exceedance probabilities
    exceedance_probs = np.exp(-0.5 * delta_bic) / np.sum(np.exp(-0.5 * delta_bic))

    # Step 4: Calculate φ (phi)
    phi = 1 / (1 + np.exp(delta_bic / 2))

    # Print results
    rounded_probs = [round(prob, 4) for prob in exceedance_probs]
    phi = [round(value, 4) for value in phi]
    print("Exceedance Probabilities:", rounded_probs)
    print("Phi Values:", phi)

def create_csv_for_matlab(data, exp):
    # create csv for matlab
    # create pivot table with pid as y and model as x and fill the values with BIC
    data = data.pivot(index="model", columns="pid", values="BIC")
    data = data.sort_index() #1743, 1756, 479, 491, 522, mb
    # data = missing_bic(data)
    data.to_csv(f"matlab/{exp}.csv", index=False, header=False)

def remove_double_mb_entries(data):
    # remove entries from the data where the model is mb and the number of parameters is 4
    data = data[~((data["model"] == "mb") & (data["number_of_parameters"] == 4))]
    # remove duplicates
    data = data.drop_duplicates(subset=["pid", "model"])
    # save as csv
    # data.to_csv(f"{exp}_{criterion}.csv", index=False)
    return data

def missing_bic(df):
    #replace the missing value by row and column average
    # Calculate row averages
    row_avg = df.mean(axis=1, skipna=True).tolist()

    # Calculate column averages
    col_avg = df.mean(axis=0, skipna=True).tolist()

    # Iterate through DataFrame
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if pd.isna(df.iat[i, j]):
                df.iat[i, j] = (row_avg[i] + col_avg[j]) / 2

    return df

if __name__ == "__main__":
    # experiment = ["v1.0", "c2.1", "c1.1"]
    experiment = ["c2.1"]
    # experiment = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
    # experiment = ["strategy_discovery"]
    df_all = []
    for exp in experiment:

        data = pd.read_csv(f"data/{exp}.csv", index_col=0)

        # add BIC
        # if exp in ["v1.0", "c1.1", "c2.1"]:
        #     data["BIC"] = compare_pseudo_likelihood(data, 35)
        # elif exp in ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]:
        #     data["BIC"] = compare_number_of_clicks_likelihood(data, 35)
        # elif exp == "strategy_discovery":
        #     data["BIC"] = compare_pseudo_likelihood(data, 120)

        if exp == "strategy_discovery":
            data["BIC"] = compare_loss(data, 120)
        else:
            data["BIC"] = compare_loss(data, 35)

        data = data[data["pid"].isin(learning_participants[exp])]
        df_all.append(data)

    result_df = pd.concat(df_all, ignore_index=True)
    # create_csv_for_matlab(result_df, "lvlc")
    model_bic = sort_by_BIC(result_df)
    # bms(model_bic)
