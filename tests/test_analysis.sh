#!/bin/sh
cd ..
#python mcl_toolbox/infer_participant_sequences.py 1 v1.0 training
#python mcl_toolbox/infer_sequences.py v1.0 35 training
python mcl_toolbox/fit_mcrl_models.py v1.0 5555 pseudo_likelihood 1 35 "{\"plotting\":True, \"optimization_params\" :{\"optimizer\":\"hyperopt\", \"num_simulations\": 2, \"max_evals\": 2}}"