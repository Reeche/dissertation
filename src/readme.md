Metacognitive Learning Tool box
==============================


https://re.is.mpg.de

What Is This?
-------------

This repository contains two modules used to analyse metacognitive learning in human.
`src/computationa_microscope` contains the code for the computational microscope
`src/mcrl_modelling` contains the code to fit the metacognitive reinforcement learning models (MCRL) to the data. 

How To Use This
---------------

### Installation
Run `src/requirements.txt` to install the dependencies. 

### Import data
Assuming you are working with the Mouselab-MDP repository and with a postgres database: 
1. Navigate to `src/import_data`
2. Put your dataclip (csv file) in the folder `src/import_data/data`
2. Run `src/import_data/reformat_csv.py` to create the required mouselab-mdp.csv and participants.csv
for each condition as well as an overall file
   
Note: you might have to use your own import code depending on your requirements. 

### Analysis modules
1. Navigate to `src/`
2. Run `src/infer_participant_sequences.py` to analyse the click sequence of each participant
3. Run `src/infer_participant.py` to analyse the click sequence average over conditions
4. Run `src/fit_mcrl_models.py` to fit the MCRL models

Note: see each folder or each file for detailed instructions. 

Testing
-------

TODO

Development
-----------

Please fork your own feature branch and merge in the dev branch. 

