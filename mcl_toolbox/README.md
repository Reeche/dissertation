# Modeling metacognitive learning mechanisms

This repository contains code for learning mechanisms implemented in the dissertation:
1. Model-based MCRL
2. LVOC and its variants
3. REINFORCE and its variants
4. RSSL

Below are the set of models with different additional cognitive mechanisms. 
For example, the plain hybrid Reinforce is Reinforce where `use_pseudo_rewards` is False, `subjective_cost` is False, and `termination_value_known` is False.
The Mental habit model and the non-learning model are special variants of the Reinforce model. 
The mental habit model is Reinforce where `is_null` is True and `features` is habitual features. 
The non-learning model is Reinforce where `is_null` is True and `features` is non-habitual features.

The model index for the used main models in the dissertation are: 
- Model-free Reinforce: 
- Hybrid Reinforce:
- Model-free LVOC: 
- Hybrid LVOC:
- Mental Habit Model:
- Non-learning Model:
- RSSL: 


| column | description | values | notes                                                                                                                 |
|---|---|---|-----------------------------------------------------------------------------------------------------------------------|
| model | Describes the general model class  | hierarchical_learner, sdss, lvoc, reinforce, rssl  |                                                                                                                       |
| decision_rule | The decision rule in the top level of the hierarchical learner that decides whether or not to plan further | threshold, adaptive_satisficing, confidence_bound  | There are more decision rules implemented but not tested yet                                                          |
| use_pseudo_rewards | Whether or not the models use pseudo rewards  |   |                                                                                                                       |
| pr_weight | Weight to the pseudo rewards  |   |                                                                                                                       |
| actor | Determines the lower level learner of the hierarchical learner  |   |                                                                                                                       |
| term | Whether or not the bottom level learner in the hierarchical learner terminates  |   |                                                                                                                       |
| selector | Determines the top level of the SDSS model  |   |                                                                                                                       |
| learner | Determines the bottom level learner of the SDSS model  |   |                                                                                                                       |
| strategy_space_type | Which strategy space to consider  | microscope (contains all strategies), participants (reduced set of most frequent strategies)  |                                                                                                                       |
| stochastic_updating | This is uses in the RSSL model to decide how to update the priors (based on a cointoss or explicit probability) |   |                                                                                                                       |
| subjective_cost | This is a parameter added to every click to incorporate additional planning costs people might have  |   |                                                                                                                       |
| vicarious_learning | updates parameters after each action using term_features and term_reward (what if I terminate now?)  |   | only applicable for LVOC                                                                                              |
| termination_value_known | Whether or not the termination value is assumed by the model  |   |                                                                                                                       |
| montecarlo_updates | do not only update after every click but at the end of trial using all clicks in that trial  |   | only applicable for LVOC                                                                                              |
| is_null | Whether or not the model learns at all  |   | If true, it doesn't update its parameters. Only the initial parameters are used.                                      |
| is_gaussian | Whether or not the RSSL model is gaussian (if not gaussian, it is binomial)  |   |                                                                                                                       |
| bandit_prior | Whether or not we fit the prior for the top level RSSL model in the SDSS learner  |   |                                                                                                                       |
| prior | This determines what kind of parameters are optimized for. strategy_weight optimizes for feature based learners and other priors optimize for rssl parameters (2 x num_strategies parameters)  |   |                                                                                                                       |
| features | Two feature sets are considered  | Habitual features (contains all features), non-habitual features (contains all features except for the 5 habitual features)  |                                                                                                                       |
| learn_from_path | Whether the model should learn from the nodes it walks upon  |   | In some paradigms, the path is revealed by walking without explicitly clicking. In this case, learn_from_path is True |

## How to fit the models
To fit the model-free models, you can use the `fit_mcrl_models.py` script.
To fit the model-free models, you can use the `fit_model_based_models.py` script.

## Check for completeness
To check whether you have fitted all the models for all the participants, you can use the `check_fitted_models.py` script.

## Simulate fitted model
To run the fitted model, you can use the `run_fitted_model.py` script and see how the model performs using the fitted hyperparameters.
To simulate the model performance using a manually defined set of hyperparameters, you can use the `simulate_model.py` script.

# Computational Microscope
To infer the model's planning strategy using the computational microscope, you can use the `infer_model_sequences.py` script.
To infer the participant's planning strategy using the computational microscope, you can use the `infer_participant_sequences.py` script.
To calculate the score and clicks of a certain strategy, you can use the `calculate_strategy_score.py` script.