This folder contains the code to read the data from the mouselab experiments.

First download the data from the database as csv. Then save it as 'dataclip.csv' and put it into the data folder.
`reformat_csv.py` will create two csv in this directory, one containing information on all participants and one
containing information on all mouselab-mdp trials. It will also create separate csv files for all three conditions and
location them in corresponding folders (`data/human/{exp}`). 