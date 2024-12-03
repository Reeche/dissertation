import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis.strategy_discovery_analysis.vars import clicked_pid, not_examined_all_pid

"""This analysis looks at the proportion of optimal strategies of the simulated models. 
For all the models that did NOT use the optimal strategy in the first trial but used it on the last trial. 
It counts the proportion.

MF: 
Not examined all pid: 0.13568985
clicked pid: 0.1327674

Hybrid: 
Not examined all pid: 0.14375695
clicked pid: 0.14460622
"""


all_reward_data = []
for pid in clicked_pid:
    try:
        data = pd.read_pickle(f"test7/{pid}_491_10.pkl")
    except FileNotFoundError:
        print(f"Could not load {pid}_491_10.pkl")
        continue

    # save them all in a list
    all_reward_data.append(data["r"])

# flatten all_reward_data
all_reward_data = [item for sublist in all_reward_data for item in sublist]

# for list in list in list, replace 13, 14, 15 with 1 and 0 otherwise
all_reward_data = [[1 if x in [13, 14, 15, 16] else 0 for x in sublist] for sublist in all_reward_data]

# remove the lists where the first entry is 1
all_reward_data_filtered = [x for x in all_reward_data if x[0] == 0]

# calculate the average proportion
average_proportion = np.mean(all_reward_data_filtered, axis=0)
print(average_proportion)

# plot the average proportion
plt.plot(average_proportion, color="b", label="Adaptive")
plt.legend()
plt.show()
plt.close()

