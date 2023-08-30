import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency

os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
stats = importr('stats')

if __name__ == "__main__":
    # experiment = "v1.0"
    experiments = ["v1.0", "c2.1", "c1.1"]
    for experiment in experiments:
        participants = pd.read_pickle(f"../../results/cm/inferred_strategies/{experiment}_training/strategies.pkl")

        # get all used strategies
        participants_df = pd.DataFrame.from_dict(participants, orient='index')
        unique_used_strategies = pd.unique(participants_df.values.flatten())

        #get all the used pairs
        pairs = []
        for key, values in participants.items():
            pairs.append(list(zip(values, values[1:])))

        all_pairs = [item for sublist in pairs for item in sublist]

        # count how often a pair was observed
        pairs_count_df = pd.DataFrame(0, index=unique_used_strategies, columns=unique_used_strategies)
        for pair in all_pairs:
            pairs_count_df[pair[0]][pair[1]] += 1

        pairs_count = pairs_count_df.to_numpy()
        pairs_count_no_diagonal = pairs_count[~np.eye(pairs_count.shape[0],dtype=bool)].reshape(pairs_count.shape[0],-1)
        res = stats.fisher_test(pairs_count_no_diagonal, simulate_p_value=True)
        # res = chi2_contingency(pairs_count_df)
        print(res)
