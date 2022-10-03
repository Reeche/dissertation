import json
import pandas as pd
import numpy as np

experiment = "strategy_discovery_pilot_v1.2"

data_full = pd.read_csv(f"data/dataclips_{experiment}.csv", sep=",")

# remove unfinished data entries
data_full["endhit"].replace("", np.nan, inplace=False)
data_full["hitid"].replace("HIT_ID", np.nan, inplace=False)
data_full.dropna(subset=["endhit"], inplace=True)
data = data_full.reset_index(drop=True)
# data.drop(index=0, inplace=True) #drops first row


data = data[["datastring"]].to_dict()  # is a df
data_value = data.get("datastring")

data_value = list(data_value.values())

for rows in data_value:
    row_dict = json.loads(rows)  # transform str into dict
    answer = row_dict["data"][-2]["trialdata"]["response"]["Q1"]
    print(answer)
