# Modeling MCRL

This repository contains code for learning mechanisms implemented for modeling how people learn how to plan.

The algorithms implemented are:

1. LVOC and its variants
2. Hierarchical LVOC and its variants
3. REINFORCE and its variants
4. RSSL
5. Hybrid model combining Strategy selection algorithm (RSSL) with Strategy Discovery algorithms (LVOC, REINFORCE)

# Directory Structure (to be adjusted)

```
+--README.md									<-- this file
+--python/									<-- python code for modelling the learning mechanisms
|  +--Experiment priors.ipynb
|  +--Info data max.ipynb
|  +--Metacognitive Learning Analysis.ipynb
|  +--Model selection.ipynb
|  +--Untitled.ipynb
|  +--Untitled1.ipynb
|  +--__init__.py
|  +--__pycache__
|  +--analysis_utils.py
|  +--base_learner.py
|  +--computational_microscope.py
|  +--data
|  +--distributions.py
|  +--experiment_utils.py
|  +--generic_mouselab.py							<-- contains mouselab environment classes derived from the OpenAI gym environment, these notably are feature-based to be used in the modelling pipeline
|  +--hierarchical_models.py
|  +--index_rl_models.py
|  +--info_run.py
|  +--info_run_max.py
|  +--learning_utils.py
|  +--lvoc_models.py
|  +--model_params.json
|  +--model_reward_verification.py
|  +--models.json
|  +--modified_mouselab.py							<-- contains TrialSequence class and node-level reward functions which are used in the GenericMouselabEnv class
|  +--optimizer.py
|  +--param_search_space.json
|  +--planning_strategies.py
|  +--reinforce_models.py
|  +--results
|  +--rl_models.csv								<-- contains all the considered RL models, columns detailed below
|  +--rssl_models.py
|  +--sdss_models.py
|  +--sequence_utils.py
|  +--strategy_score.py
|  +--utils.py
|  +--results/
|  |  +--v1.0_plots/
|  +--data/									<-- outputted modelling data for each experiment
|  |  +--DS_proportions.pkl
|  |  +--L1_DS.pkl
|  |  +--L1_distances.pkl
|  |  +--L1_norm_DS.pkl
|  |  +--L2_DS.pkl
|  |  +--L2_distances.pkl
|  |  +--L2_norm_DS.pkl
|  |  +--cluster_confusions_bernoulli_rssl.pkl
|  |  +--cluster_confusions_gradual.pkl
|  |  +--cluster_confusions_mixed.pkl
|  |  +--cluster_confusions_random.pkl
|  |  +--cluster_confusions_random_switch.pkl
|  |  +--cluster_confusions_strategy_only.pkl
|  |  +--cluster_scores.pkl
|  |  +--decision_system_features.pkl
|  |  +--decision_systems.pkl
|  |  +--ds_validation_sequences.pkl
|  |  +--em_features.pkl
|  |  +--exp_pipelines.pkl
|  |  +--exp_reward_structures.pkl
|  |  +--feature_systems.pkl
|  |  +--implemented_features.pkl						<-- feature priors, containing all considered features
|  |  +--jeffreys_divergences.pkl
|  |  +--js_divergences.pkl
|  |  +--kl_cluster_map.pkl
|  |  +--kl_clusters.pkl
|  |  +--microscope_features.pkl						<-- feature priors, missing a few features #TODOCUMENT
|  |  +--microscope_weights.pkl
|  |  +--new_nn_features.pkl
|  |  +--nn_features.pkl
|  |  +--non_problematic_clusters.pkl
|  |  +--normalized_values
|  |  +--prior_transitions_v1.pkl
|  |  +--rssl_participant_priors.pkl
|  |  +--same_strategy_validation.pkl
|  |  +--strategy_confusions_bernoulli_rssl.pkl
|  |  +--strategy_confusions_gradual.pkl
|  |  +--strategy_confusions_mixed.pkl
|  |  +--strategy_confusions_random.pkl
|  |  +--strategy_confusions_random_switch.pkl
|  |  +--strategy_confusions_strategy_only.pkl
|  |  +--strategy_decision_proportions.pkl
|  |  +--strategy_decision_weights.pkl
|  |  +--strategy_feature_scores.pkl
|  |  +--strategy_scores.pkl
|  |  +--strategy_space.pkl
|  |  +--strategy_validation_sequences.pkl
|  |  +--normalized_values/
|  |  |  +--nn/
|  |  |  +--high_increasing/
|  |  |  +--high_decreasing/
|  |  |  +--v1.0/
|  |  |  +--c2.1/
|  |  |  +--T1.1/
|  |  |  +--3_1_1_2_3/
|  |  |  +--low_constant/
|  |  |  +--c1.1/
|  |  |  +--large_increasing/
|  |  |  +--F1/
|  |  |  +--3_1_2/
+--data/									<-- data for various mouselab experiments
|  +--human/									<-- human data (as opposed to simulated data)
|  |  +--v1.0/									<--
|  |  +--c2.1/									<--
|  |  +--T1.1/									<--
|  |  +--c1.1/									<--
|  |  +--F1/									<-- data for the experiment with increasing variance and 3-1-2 branching
```

# Considered Reinforcement Learning Models

| column | description | values | notes |
|---|---|---|---|
| model |   |   |   |
| decision_rule |   |   |   |
| use_pseudo_rewards |   |   |   |
| pr_weight |   |   |   |
| actor |   |   |   |
| term |   |   |   |
| selector |   |   |   |
| learner |   |   |   |
| strategy_space_type |   |   |   |
| stochastic_updating |   |   |   |
| subjective_cost |   |   |   |
| vicarious_learning |   |   |   |
| termination_value_known |   |   |   |
| montecarlo_updates |   |   |   |
| is_null |   |   |   |
| is_gaussian |   |   |   |
| bandit_prior |   |   |   |
| prior |   |   |   |