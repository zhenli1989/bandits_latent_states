# coding: utf-8
import os

from joblib import load, dump

from src.utils import INIT_DISTRIBUTION

####################################################################################
####################################################################################
############################# AREA OF INPUT PARAMETERS #############################
####################################################################################
####################################################################################
##### Environment Parameters
PATH = "." # Root directory, should be the same path this "README.md" file locates
PATH_DATA = f"{PATH}/data" # Path for data
PATH_MODELS = f"{PATH}/models"  # Path for models

##### Parameters for Bandits
number_rounds = 50000 # T, use 50000 to reproduce the results
list_random_seed = list(range(1951, 2051)) # To reproduce the results, use 1951 to 2051

####################################################################################
####################################################################################
############# Create Output Path, Load the data and Model###############
####################################################################################
####################################################################################
##### Load Data and Model
for random_seed in list_random_seed:
    print(random_seed)
    if os.path.isfile(
            f"{PATH_MODELS}/random_seed_{random_seed}/list_belief_estimated.pkl"):
        print('skip')
        continue
        print("------------------------------")

    Spectral_direct_estimator = load(f"{PATH_MODELS}/random_seed_{random_seed}/spectral_direct_estimator.pkl")
    list_belief_estimated = []
    for curr_iter_ in range(1, number_rounds + 1):
        hist_belief_ = Spectral_direct_estimator.update_belief_one_iteration(vec_init=INIT_DISTRIBUTION, number_round=curr_iter_)
        list_belief_estimated.append( hist_belief_[-1].copy())

    dump(list_belief_estimated,f"{PATH_MODELS}/random_seed_{random_seed}/list_belief_estimated.pkl")
    print("------------------------------")
