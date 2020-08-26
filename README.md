# Modeling MCRL
This repository contains code for learning mechanisms implemented for modeling how people learn how to plan.

The algorithms implemented are:
1. LVOC and its variants
2. Hierarchical LVOC and its variants
3. REINFORCE and its variants
4. RSSL
5. Hybrid model combining Strategy selection algorithm (RSSL) with Strategy Discovery algorithms (LVOC, REINFORCE)

# Directory Structure

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
|  +--generic_mouselab.py
|  +--hierarchical_models.py
|  +--index_rl_models.py
|  +--info_run.py
|  +--info_run_max.py
|  +--learning_utils.py
|  +--lvoc_models.py
|  +--model_params.json
|  +--model_reward_verification.py
|  +--models.json
|  +--modified_mouselab.py
|  +--optimizer.py
|  +--param_search_space.json
|  +--planning_strategies.py
|  +--reinforce_models.py
|  +--results
|  +--rl_models.csv
|  +--rssl_models.py
|  +--sdss_models.py
|  +--sequence_utils.py
|  +--strategy_score.py
|  +--utils.py
|  +--results/
|  |  +--v1.0_plots/
|  +--data/									<-- outputted modelling data for each experiment
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