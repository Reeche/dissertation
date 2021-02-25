import os
import statistics

import numpy as np
import pandas as pd

score_list = {}

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pkl") and file.startswith("c1.1"):
            #print(os.path.join(root, file))
            object = pd.read_pickle(file)
            object_float = np.array(object[0], dtype="float32")
            #print(statistics.mean(object_float))
            score_list.update({file: statistics.mean(object_float)})

print(dict(sorted(score_list.items(), key=lambda item: item[1], reverse=True))) #set to true for best ones; false for the worst ones