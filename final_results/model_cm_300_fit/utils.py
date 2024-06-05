import pandas as pd
import glob

def fill_model_column(df):
    # if model column is empty, take the value from model_index and fill it into the model column
    df['model'] = df.apply(lambda x: x['model_index'] if pd.isnull(x['model']) else x['model'], axis=1)
    # remove the model_index and class columns
    df = df.drop(columns=['model_index', 'class'])
    return df

# do it for all csv files in this dir
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    try:
        df = fill_model_column(df)
        df.to_csv(file, index=False)
    except Exception as e:
        print(f"Error in file {file}: {e}")
        continue