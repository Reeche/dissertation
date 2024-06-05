import os
import pandas as pd

# conditions = ["high_variance_high_cost", "high_variance_low_cost", "low_variance_high_cost", "low_variance_low_cost"]
# conditions = ["v1.0", "c2.1", "c1.1",
#               "high_variance_high_cost", "high_variance_low_cost",
#               "low_variance_high_cost", "low_variance_low_cost",
#               "strategy_discovery"]
conditions = ["high_variance_high_cost", "high_variance_low_cost",
               "low_variance_high_cost", "low_variance_low_cost"]

for condition in conditions:
    # look for csv files with the same ending as condition
    csv_files = [f for f in os.listdir() if f.endswith(f"{condition}.csv")]
    # csv_files = [f for f in os.listdir() if f.startswith(f"{condition}")] #for cm
    # merge the csv files
    df = pd.concat([pd.read_csv(f) for f in csv_files])
    # save the merged csv file
    df.to_csv(f"{condition}.csv")
