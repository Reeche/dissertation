import numpy as np
import pandas as pd

df = pd.read_csv("participants-mf_stroop_full_exp.csv", sep=",")
df1 = pd.read_csv("participants-mf_stroop_full_exp.csv", sep=",")
# append df
df = df._append(df1, ignore_index=True)

# filter for condition = 0
# df = df[df["condition"] == 1]

print("Number of participants", len(df))
gender = df["gender"].value_counts()
print(gender)

age = np.mean(df["age"])
min_age = df["age"].min()
max_age = df["age"].max()
print(age, min_age, max_age)

bonus = np.mean(df["bonus"])
print("average bonus", bonus)