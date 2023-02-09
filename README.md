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
To install as a package in python 3.8+:
```
git clone https://github.com/RationalityEnhancement/mcl_toolbox.git
pip install -e .
cd mcl_toolbox
```

The process of installing the dependency ```torch``` varies by system. While the rest of the dependencies install automatically, PyTorch will have to be installed manually from the [PyTorch website](https://pytorch.org/get-started/locally/).

### Import data
Assuming you are working with the Mouselab-MDP repository and with a postgres database: 
1. Navigate to `src/import_data`
2. Put your dataclip (csv file) in the folder `src/import_data/data`
2. Run `src/import_data/reformat_csv.py` to create the required mouselab-mdp.csv and participants.csv
for each condition as well as an overall file
   
Note: you might have to use your own import code depending on your requirements. 

### Analysis modules
1. Navigate to `mcl_toolbox/`
2. Run `python mcl_toolbox/infer_participant_sequences.py` to analyse the click sequence of each participant
3. Run `python mcl_toolbox/infer_sequences.py` to analyse the click sequence average over conditions
4. Run `python mcl_toolbox/fit_mcrl_models.py` to fit the MCRL models

Note: see each folder or each file for detailed instructions. 

Testing
-------
There are very simple integration tests in tests/ to run analysis modules quickly to check whether analysis modules will run.
To run these, run:
```
chmod +x test_analysis.sh
./test_analysis.sh
```

CogSci analysis 
-------
The folder cogsci_analysis contains the analysis conducted for the CogSci submission 2023. 
Please contact the author for the resulting pkl files of the model fitting results. 
* create_df.py is used to create intermediate csv files that can be found in cogsci_analysis/results 
* model_selection_analysis.py calculates BIC, AIC 
* model_recovery_analysis.py performs model recovery

Development
-----------

Please fork your own feature branch and merge in the dev branch. 

