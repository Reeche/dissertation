import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
"""
some ad hoc analysis
* Check how many participants answered the questions correctly
* check the correlation between score and questions correctly answers

"""
data = pd.read_csv("mcl_toolbox/import_data/data/dataclips.csv", sep=',')

# remove unfinished data entries
data['endhit'].replace('', np.nan, inplace=False)
data['hitid'].replace('HIT_ID', np.nan, inplace=False)
data.dropna(subset=['endhit'], inplace=True)
data = data.reset_index(drop=True)

original_data = data[data["cond"] == 0]  # filter for condition 0, where participants got to see the examples

data = original_data["datastring"]
status = original_data["status"]
data = data.to_dict()
for key, value in data.items():
    data[key] = json.loads(value)

# # add status, either 4 or 3
# for key, value in data.items():
#     data[key]["status"] = status[key]

df = pd.DataFrame(columns=["Correct answers", "Bonus"])

# loop through PID and finds the correctly answered questions and corresponding bonus
correct_answers = []
bonus = []
pids = list(data.keys())
for pid in pids:
    if data[pid]["currenttrial"] == 49:
        correct_answers.append(sum(data[pid]['data'][14]['trialdata']['correct']))
        try:
            bonus.append(data[pid]['questiondata']['final_bonus'])
        except:
            bonus.append(0)
            continue
    elif data[pid]["currenttrial"] == 51:
        correct_answers.append(sum(data[pid]['data'][16]['trialdata']['correct']))
        try:
            bonus.append(data[pid]['questiondata']['final_bonus'])
        except:
            bonus.append(0)
            continue
    else:
        correct_answers.append(sum(data[pid]['data'][18]['trialdata']['correct']))
        try:
            bonus.append(data[pid]['questiondata']['final_bonus'])
        except:
            bonus.append(0)
            continue

df["Correct answers"] = correct_answers
df["Bonus"] = bonus
print(df)

stat, p = stats.spearmanr(df["Correct answers"], df["Bonus"])
print("p-value", stat, p)

plt.plot(df["Correct answers"], df["Bonus"], 'ro')
plt.xlabel('Correct answers')
plt.ylabel('Bonus')
plt.show()