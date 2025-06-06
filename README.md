Code for the dissertation: Metacognitive Reinforcement Learning: a theory of how people learn how to plan
==============================

What Is This?
-------------

This repository contains the code for the dissertation "Metacognitive Reinforcement Learning: a theory of how people learn how to plan" by Ruiqi He. 
It contains three main components:
1. `mcl_toolbox/computationa_microscope` contains The computational microscope, which is a tool to analyse the click sequences of human participants in the Mouselab-MDP task.
2. `mcl_toolbox/models` contains the models, which are used to fit the click sequences of human participants in the Mouselab-MDP task.
3. `analysis` contains the analyses scripts for the analyses presented in the dissertation. 

The first two directories contain a README file with detailed instructions on how to use the code.

How To Use This
---------------

### Installation
Run requirements.txt to install the dependencies:

```
pip install -r requirements.txt
```


Testing
-------
There are very simple integration tests in tests/ to run analysis modules quickly to check whether analysis modules will run.
To run these, run:
```
chmod +x test_analysis.sh
./test_analysis.sh
```

Examples
-------
The folder `notebooks/` contains example scripts for how to use the Computational Microscope and the models.

