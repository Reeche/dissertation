An example analysis of an experiment using sequences obtained using our Computational Microscope is done in experiment-analysis.ipynb

For running the computational microscope on your dataset (Suggested):
  1. Put the dataset into the data/human directory of the repo and give the folder the name <exp_num>
  2. The data has to be in the mouselab-mdp.csv file with the keys "stateRewards", "queries", "pid" and "block"

This file assumes that you are using one of the earlier experiments or using the 3-1-2 structure with increasing variance in rewards.

To run the computational microscope on the data for a particular pid:

  `python3 infer_participant_sequences.py <pid> <exp_num> [<block>]`

The results are stored in `python/results/inferred_sequences/<exp_num>`

<exp_num> can currently take values "F1", "v1.0", "T1.1" and "c1.1"
