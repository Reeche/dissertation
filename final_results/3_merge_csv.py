import os
import pandas as pd

# conditions = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
conditions = ["v1.0", "c2.1", "c1.1"]

for condition in conditions:
    # look for csv files with the same ending as condition
    csv_files = [f for f in os.listdir() if f.endswith(f"{condition}.csv")]
    # merge the csv files
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    # save the merged csv file
    df.to_csv(f"{condition}.csv")