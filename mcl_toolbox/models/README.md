# List of MCRL models (19 February 2023)

models.csv contains the list of MCRL models. The old_index is the index used in previous modelling experiments. 
The value in the curly brackets indicate the values that it can take for the corresponding model. 
For example, for Reinforce models vicarious_learning is always FALSE, while for LVOC models it can be TRUE or FALSE. 

| Model class | decision rule | use_pseudo_rewards | pr_weights | actor | term | selector | learner | strategy_space| stochastic_updating | subjective_cost | vicarious_learning | termination_value_known  | monte_carlo_updates | is_gaussian | bandit_prior | prior | habitual_features | learn_from_path |
| --- | --- | --- | --- | --- | --- |---| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (Baseline) Reinforce | na | True, False | if pseudo_reward is False, this is always false; otherwise True, False | na | True | na | na | na | na | True, False | False | True, False | False | na | na | strategy_weights | habitual, non-habitual | True, False |
| LVOC | na | True, False | same as Reinforce | na | True | na | na | na | na | True, False | True, False | True, False | True, False | na | na | strategy_weights | habitual, non-habitual | True, False |
| Hierarchical | adaptive_satisficing, confidence_bound, threshold | True, False | same as Reinforce | LVOC, Reinforce, Baseline Reinforce | True, False | na | na | na | na | True, False | depends on actor | True, False | depends on actor | na | na | strategy_weights | habitual, non-habitual | True, False |
| RSSL | na | True, False | same as Reinforce | na | na |na| na | microscope, participants | True, False | na | na | na | na | True, False | na | gaussian_prior, bernoulli_prior | habitual, non-habitual | na |
| SDSS | na | True, False | same as Reinforce | True, False | True | rssl | LVOC, Reinforce, Baseline reinforce | na | True, False | True, False | depends on learner | True, False | depends on learner | True, False | True | strategy_weights | habitual, non-habitual | True, False |

### Non-learning models
Next to the learning models, there are 44 non-learning models (is_null = True)

| Model class | decision rule | use_pseudo_rewards | pr_weights | actor | term | selector | learner | strategy_space| stochastic_updating | subjective_cost | vicarious_learning | termination_value_known  | monte_carlo_updates | is_gaussian | bandit_prior | prior | habitual_features | learn_from_path |
| --- | --- | --- | --- | --- | --- |---| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all above | na or all hierarchical rules | False | False | na or all actors above | True, False | na or rssl | na or all models | participants | False | False | False | False | False | na or False | na or True | bernoulli or strategy_weights | habitual, non-habitual | False |
