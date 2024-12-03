import pandas as pd
import numpy as np
from vars import clicked_pid, assign_model_names, adaptive_pid, mod_adaptive_pid, maladaptive_pid

"""
This file creates csv for BMS analysis in Matlab
It does not contain the model-based models
"""

def compare_loss(data, trials):
    BIC = 2 * data["loss"] + data["number_of_parameters"] * np.log(trials)
    return BIC
def create_csv_for_matlab(data, exp):
    data = data.pivot(index="model", columns="pid", values="BIC").T
    data = data[['Habitual', 'Non-learning', 'SSL', 'hybrid Reinforce', 'MF - Reinforce']]
    data.to_csv(f"cogsci_matlab/{exp}_maladaptive.csv", index=False, header=False)


data = pd.read_csv(f"../../final_results/aggregated_data/strategy_discovery.csv", index_col=0)

data = data[data["pid"].isin(maladaptive_pid)]

data['model'] = data.apply(assign_model_names, axis=1)
data["BIC"] = compare_loss(data, 120)

create_csv_for_matlab(data, "strategy_discovery")