import os
from pathlib import Path

from mcl_toolbox.env.generic_mouselab import GenericMouselabEnv
from mcl_toolbox.global_vars import features, model, pickle_load, strategies
from mcl_toolbox.utils.experiment_utils import Experiment

implemented_features = features.implemented
microscope_features = features.microscope
strategy_weights = strategies.strategy_weights
model_attributes = model.model_attributes
strategy_spaces = strategies.strategy_spaces


def construct_model(model_index, num_actions, normalized_features):
    learner_attributes = model_attributes.iloc[model_index].to_dict()
    learner = learner_attributes["model"]
    strategy_space_type = learner_attributes["strategy_space_type"]
    strategy_space_type = strategy_space_type if strategy_space_type else "microscope"
    strategy_space = strategy_spaces[strategy_space_type]
    if learner_attributes["habitual_features"] == "habitual":
        features = implemented_features
    else:
        features = microscope_features
    if learner == "rssl":
        num_priors = 2 * len(strategy_space)
    else:
        num_priors = len(features)
    pr_weight = learner_attributes["pr_weight"]
    if not pr_weight:
        learner_attributes["pr_weight"] = 1
    learner_attributes = dict(
        features=features,
        normalized_features=normalized_features,
        num_priors=num_priors,
        strategy_space=strategy_space,
        no_term=not learner_attributes["term"],
        num_actions=num_actions,
        **learner_attributes,
    )
    del learner_attributes["term"]
    return learner, learner_attributes


def get_participant_context(exp_num, pid, pipeline, exp_attributes={}):
    E = Experiment(exp_num, **exp_attributes)
    E.attach_pipeline(pipeline)
    participant = E.participants[pid]
    q_fn = None
    file_location = Path(__file__).parents[1]
    if "condition" in dir(participant):
        if participant.condition == "meta":
            try:
                q_path = os.path.join(file_location, f"data/{exp_num}_q.pkl")
                q_fn = pickle_load(q_path)["q_dictionary"]
            except FileNotFoundError:
                print("Q-fn could not be loaded")
    else:
        participant.condition = "none"
    env = GenericMouselabEnv(
        len(participant.envs),
        pipeline=pipeline,
        ground_truth=participant.envs,
        feedback=participant.condition,
        q_fn=q_fn,
    )
    return participant, env
