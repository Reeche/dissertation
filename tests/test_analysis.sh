#!/bin/sh
cd ..
python mcl_toolbox/infer_participant_sequences.py 1 F1 training
python mcl_toolbox/infer_sequences.py F1 training
python mcl_toolbox/fit_mcrl_models.py F1 1729 pseudo_likelihood 1 "{\"plotting\":True, \"optimization_params\" : {\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"