This folder contains the python code that implements various learning algorithms that try to model how people learn how
to plan.

These models can be applied to the same trials that participants saw or custom trials designed by the researcher.

We have a model space of 6431 learning models with different attributes. The properties of the models are present in the
file rl_models.csv

### Models
Files that implement the various types of learning algorithms:

1. `base_learner.py` - Contains the base class for all the learning algorithms
2. `lvoc_models.py` - Implements the Learned Value of Computation (LVOC) model and its variants.
3. `reinforce_models.py` - Implements the Vanilla REINFORCE model and the baseline version of the REINFORCE model
4. `rssl_models.py` - Implements the Rational Strategy Selection Learner (RSSL) model.
5. `sdss_models.py` - Implements the Strategy Disovery and Strategy Selection (SDSS) model. This model combines the RSSL
   model with the LVOC model. The RSSL model selects the strategies that are learned from reward.
6. `hierarchical_models.py` - Implements a two-stage model, where the first stage decides whether or not to terminate
   planning and the second stage is one of the variants of the LVOC or the REINFORCE models.

### How to run
To run the model <model_index> on the same trials of the participant <pid> of the experiment <exp_num>, run the
following command:
`python3 index_rl_models.py <model_index> <exp_num> <optimization_criterion> <pid>`
where <optimization_criterion> is one
of ["likelihood","pseudo_likelihood", "mer_performance_error", "performance_error", "clicks_overlap", "number_of_clicks_likelihood"]. Other optimization criteria
can be found in `compute_objective` function in the file `learning_utils.py`

###Optimization criterions:
1. `pseudo_likelihood` - uses the MER of the participant as it's mean and the likelihood is computed using the algorithm's max expected return
2. `mer_performance_error` - MER is the maximum expected return (!= reward/score). Instead of using the actual scores, we use the max expected returns (i.e expected return before moving on the path) to compute the error
3. `performance_error` - the sum of squared errors where each error is computed as the difference in the score obtained by the algorithm and the participants
4. `clicks_overlap` - proportion of clicks made by the learning algorithm that were in common with the participants' clicks
5. `number_of_clicks_likelihood` - Optimises for the likelihood of number of clicks
6. `likelihood` - computes the click level likelihood for the models, which would be ideal but the problem with it was that it couldn’t be computed for our hybrid models. So, to maintain uniformity we use the pseudo-likelihood

The criterions without likelihood are not direct likelihood functions but some kind of a summary statistic over the rewards obtained by the algorithms and the participants.
We convert the summary statistic to a likelihood function by assuming a normal distribution with mean as the participant reward vector and a variance parameter that is to be fit along with other parameters of the model

Note: 
In `optimizer.py` pyabc is negated because pyabc takes in a distance fn as input and distances are positive and lower number mean better fit
Since the domain of let's say reward is the space or real numbers, unless we use some kind of normalization, we can't make it positive all the time (in a general way).
The compute_objective function in learning_utils.py is negated for some criteria because reward, strategy accuracy, clicks_overlap, likelihood, pseudo-likelihood are naturally objectives we want to maximize, but to be compatible with hyperopt which only minimizes, we negate them.
Currently, pyabc is not able to optimise for these criteria: "reward", "strategy_accuracy", "clicks_overlap"

### Model features
