import numpy as np
import pandas as pd

df = pd.read_csv("participants-all_planningamount.csv", sep=",")
print("Number of participants", len(df))
gender = df["gender"].value_counts()
print(gender)

age = np.mean(df["age"])
min_age = df["age"].min()
max_age = df["age"].max()
print(age, min_age, max_age)

bonus = np.mean(df["bonus"])
print("average bonus", bonus)