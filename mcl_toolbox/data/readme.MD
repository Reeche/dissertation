Overview of the pickles (todo: make it look nicer)

* normalized_values/ folder contains the max and min values of the features in different reward structures

* strategy_scores.pkl contains the average score of each strategy on the increasing variance environment

* cluster_scores.pkl contains the average score of each strategy type on the increasing variance environment
  
* strategy_space.pkl contains the list of all 79 strategies

* strategy_confusions_.pkl files are the validation results (Confusion matrix) of the Computational Microscope on the data generated from different learning models. Same goes for cluster_confusions_.pkl files

* strategy_validation_sequences.pkl contains the validation sequences for all learning models

* same_strategy_validation.pkl contains the validation sequnces for data generated using a single strategy in all the trials

* ds_validation_sequences.pkl contains sequences generated to validate the trajectories of decision systems (refer to the article on Computational Microscope)

* js_divergences.pkl contains the Jensen-Shannon Divergence between the 89 strategies (according to planning_strategies.py)

* jeffreys_divergences.pkl contains the Jeffrey Divergence or the symmetric KL divergence between the 89 strategies (according to planning_strategies.py)

* kl_clusters.pkl contains the cluster number of the strategy indexed according to strategy_space.pkl

* kl_cluster_map.pkl contains the cluster number of a given strategy in strategy_space.pkl

* exp_pipelines.pkl contains the pipeline (reward structure for each trial) for a given experiment in the mcl_toolbox/data folder

* DS_proportions.pkl contains the proportion of each decision systems for each strategy

* decision_system_features.pkl contains the list of features belonging to each decision system

* decision_systems.pkl contains the list of decision systems

* implemented_features.pkl contains the list of all implemented features

* em_features.pkl contains the list of features used for Automatic Strategy Discovery (Expectation Maximization for Interpretable human planning)

* exp_reward_structures.pkl maps an experiment to its reward structure

* feature_systems.pkl maps a feature to its decision system

* microscope_features.pkl contains the list of features used by the Computational Microscope. The computational microscope only uses the subset of features that are independent of what the participant did on the previous trials, i.e. the habitual features are left out

* microscope_weights.pkl contains the weights of each strategy in terms of microscope_features

* same_strategy_validation.pkl contains sequences generated using a single strategy in all the trials and the inferences made by the Computational Microscope on it.

* {norm_type}{representation_type}.pkl contains the norm type norm of weights of strategies represented in the representation type
where norm_type is one of L1 norm and L2 norm and
representation type: 
DS - Weights of cecision systems
norm_DS - normalized weights of decision systems
distances - weights of strategies in terms of implemented_features.pkl

The files prior_transitions_v1.pkl, non_problematic_clusters.pkl can be removed or ignored
