import pandas as pd

# open data/{exp}.csv and replace the na values with 3
# for exp in [#'v1.0',
#             #'c2.1',
#             #'c1.1',
#             'high_variance_high_cost',
#             'high_variance_low_cost',
#             'low_variance_high_cost',
#             'low_variance_low_cost',
#             ]:
#     data = pd.read_csv(f"data/{exp}.csv")
#     # for the column "number_of_parameters", replace na with 3
#     data["number_of_parameters"].fillna(3, inplace=True)
#     data.to_csv(f"data/{exp}_new.csv", index=False)


# open csv ending with strategy_discovery and merge them all into one csv
import os

# Directory where the CSV files are located
directory_path = os.getcwd()

# List all files in the directory
all_files = os.listdir(directory_path)

# Filter for files that end with 'strategy_discovery.csv'
csv_files = [f for f in all_files if f.endswith('strategy_discovery.csv')]

# Initialize an empty list to hold dataframes
dfs = []

# Iterate through the list of filtered CSV files and append their content
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes into one
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a new CSV file
output_path = os.path.join(directory_path, 'strategy_discovery.csv')
merged_df.to_csv(output_path, index=False)

print(f"All files have been merged into {output_path}")
