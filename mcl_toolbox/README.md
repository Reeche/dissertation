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
| model | Describes the general model class  | hierarchical_learner, sdss, lvoc, reinforce, rssl  |   |
| decision_rule | The decision rule in the top level of the hierarchical learner that decides whether or not to plan further | threshold, adaptive_satisficing, confidence_bound  |   |
| use_pseudo_rewards | Whether or not the models use pseudo rewards  |   |   |
| pr_weight | Weight to the pseudo rewards  |   |   |
| actor | Determines the lower level learner of the hierarchical learner  |   |   |
| term | Whether or not the bottom level learner in the hierarchical learner terminates  |   |   |
| selector | Determines the top level of the SDSS model  |   |   |
| learner | Determines the bottom level learner of the SDSS model  |   |   |
| strategy_space_type | Which strategy space to consider (one with all strategies and one with reduced set of most frequent strategies)  |   |   |
| stochastic_updating | This is uses in the RSSL model to decide how to update the priors (based on a cointoss or explicit probability) |   |   |
| subjective_cost | This is a parameter added to every click to incorporate additional planning costs people might have  |   |   |
| vicarious_learning | Whether or not the model does vicarious learning  |   |   |
| termination_value_known | Whether or not the termination value is assumed by the model  |   |   |
| montecarlo_updates | Whether or not the model performs monte-carlo updates at the end of a trial  |   |   |
| is_null | Whether or not the model learns at all  |   | If true, it doesn't update its parameters. Only the initial parameters are used.  |
| is_gaussian | Whether or not the RSSL model is gaussian (if not gaussian, it is binomial)  |   |   |
| bandit_prior | Whether or not we fit the prior for the top level RSSL model in the SDSS learner  |   |   |
| prior | This determines what kind of parameters are optimized for. strategy_weight optimizes for feature based learners and other priors optimize for rssl parameters (2 x num_strategies parameters)  |   |   |
| features | Two feature sets are considered. One without habitual features and one with habitual features  |   |   |
