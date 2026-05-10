# coding: utf-8
import os

import pandas as pd
from joblib import load, dump

from src.Linear_Bandits_Belief import LinearBanditsBelief
from src.utils import INIT_DISTRIBUTION, format_lmd

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
list_random_seed = list(range(1951, 2051)) # To reproduce the results, use use 1951 to 2051
list_ucb_multipler = [0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5]
list_lmd = [1e-9, 0.001, 0.01, 0.1, 1, 10, 100]
list_batch = [1, 15, 37, 224]
with_union_bound = False
prefix_union_bound = '_ub' if with_union_bound else ''
####################################################################################
####################################################################################
############# Create Output Path, Load the data and Model###############
####################################################################################
####################################################################################
##### Load Data and Model
for random_seed in list_random_seed:
    dt_reward_belief = pd.read_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_belief.pq")
    dt_reward_dummy = pd.read_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_dummy.pq")
    dict_kpi_rewards = load(f"{PATH_MODELS}/random_seed_{random_seed}/dict_kpi_rewards.pkl")
    list_belief_estimated = load(f"{PATH_MODELS}/random_seed_{random_seed}/list_belief_estimated.pkl")
    ##### Create the folder for the output model
    if os.path.isdir(f"{PATH_MODELS}/random_seed_{random_seed}"):
        pass
    else:
        os.makedirs(f"{PATH_MODELS}/random_seed_{random_seed}")

    ####################################################################################
    ####################################################################################
    ######### Simulate The results for the classical linear bandits ##############
    ####################################################################################
    ####################################################################################
    for batch_len_ in list_batch:
        print(batch_len_)
        for UCB_multiply_ in list_ucb_multipler:
            UCB_multiply_str_ = "0" + '{:f}'.format(UCB_multiply_).rstrip('0').rstrip('.')[2:]
            print(UCB_multiply_str_)
            for lmd_ in list_lmd:
                lmd_str_ = format_lmd(lmd_)
                print(lmd_str_)
                if os.path.isfile(
                        f"{PATH_MODELS}/random_seed_{random_seed}/BoxA_B{batch_len_}_C{UCB_multiply_str_}_Lmd{lmd_str_}{prefix_union_bound}.pkl"):
                    print('skip')
                    continue
                    print("------------------------------")
                dict_linear_bandits_ = \
                    {"seed": random_seed, "number_rounds": number_rounds, "lmd": lmd_, "simple_bound":False,
                     "vec_init": INIT_DISTRIBUTION, "batch_lengh": batch_len_, "verbose": False,
                     "UCB_multiply": UCB_multiply_, "union_bound": with_union_bound, "hot_start": 250}

                obj_linear_bandits_ = LinearBanditsBelief(dict_linear_bandits_, dt_reward_belief, dt_reward_dummy,
                                                          list_belief_estimated)
                obj_linear_bandits_.run_simulation()
                obj_linear_bandits_.clear_for_dump()
                dump(obj_linear_bandits_,
                     f"{PATH_MODELS}/random_seed_{random_seed}/BoxA_B{batch_len_}_C{UCB_multiply_str_}_Lmd{lmd_str_}{prefix_union_bound}.pkl")
                print("------------------------------")
            print('---------------------------------------------')
        print('---------------------------------------------')
        print('---------------------------------------------')