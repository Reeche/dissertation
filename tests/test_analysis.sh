#!/bin/sh
cd ..
python mcl_toolbox/infer_participant_sequences.py
python mcl_toolbox/infer_sequences.py increasing_variance training none
python mcl_toolbox/fit_mcrl_models.py 1 v1.0 pseudo_likelihood 1