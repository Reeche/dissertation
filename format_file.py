import pandas as pd

df = pd.read_csv("high_variance_high_cost_bic.csv")
df = df.reset_index()
print(df)

# df["pid"] = df.index

print(df)

df = df.pivot_table(index=["pid"],columns=["model"],values=["BIC"], fill_value=0).reset_index()
print(df)
df.to_csv("matlab_compartible_bic.csv")
