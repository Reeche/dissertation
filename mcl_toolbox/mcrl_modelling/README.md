This folder contains the python code that implements various learning algorithms that try to model how people learn how
to plan.

These models can be applied to the same trials that participants saw or custom trials designed by the researcher.

We have a model space of 3216 learning models with different attributes. The properties of the models are present in the
file rl_models.csv

To run the model <model_index> on the same trials of the participant <pid> of the experiment <exp_num>, run the
following command:
`python3 index_rl_models.py <model_index> <exp_num> <optimization_criterion> <pid>`
where <optimization_criterion> is one
of ["pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap"]. Other optimization criteria
can be found in `compute_objective` function in the file `learning_utils.py`

Files that implement the various types of learning algorithms:

1. `base_learner.py` - Contains the base class for all the learning algorithms
2. `lvoc_models.py` - Implements the Learned Value of Computation (LVOC) model and its variants.
3. `reinforce_models.py` - Implements the Vanilla REINFORCE model and the baseline version of the REINFORCE model
4. `rssl_models.py` - Implements the Rational Strategy Selection Learner (RSSL) model.
5. `sdss_models.py` - Implements the Strategy Disovery and Strategy Selection (SDSS) model. This model combines the RSSL
   model with the LVOC model. The RSSL model selects the strategies that are learned from reward.
6. `hierarchical_models.py` - Implements a two-stage model, where the first stage decides whether or not to terminate
   planning and the second stage is one of the variants of the LVOC or the REINFORCE models.