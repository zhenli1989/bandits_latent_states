# coding: utf-8
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import dump, load

from src import HMMForward, HMMBeliefForward, HMMSpectralDirectEstimation
from src.utils import transform_dt_dummy, INIT_DISTRIBUTION_STATIONARY, TRANSITION, EMISSION

####################################################################################
####################################################################################
############################# AREA OF INPUT PARAMETERS #############################
####################################################################################
####################################################################################

##### Environment Parameters
PATH = "." # Root directory, should be the same path this "README.md" file locates
PATH_DATA = f"{PATH}/data" # Path for data
PATH_MODELS = f"{PATH}/models" # Path for models

##### Parameters for Bandits
number_rounds = 50000 # T, use 50000 to reproduce the results
list_random_seed = list(range(1951, 2051)) # To reproduce the results, use 1951 to 2051
list_actions = ['no_action', 'email', 'call']

noise_std = 0.2
for random_seed in list_random_seed:
    ##### Create the folder for the output model
    if os.path.isdir(f"{PATH_MODELS}/random_seed_{random_seed}"):
        pass
    else:
        os.makedirs(f"{PATH_MODELS}/random_seed_{random_seed}")

    ####################################################################################
    ####################################################################################
    #################### Simulate the contexts and beliefs #############################
    ####################################################################################
    ####################################################################################
    dt = pd.read_parquet(f"{PATH_MODELS}/dt_ref.pq")
    model_reward_rally = load(f"{PATH_MODELS}/model_reward_rally.pkl")
    model_reward_down = load(f"{PATH_MODELS}/model_reward_down.pkl")

    HMM_Simulator = HMMForward(vec_init=INIT_DISTRIBUTION_STATIONARY, transition_matrix=TRANSITION,
                               emission_matrix=EMISSION, random_seed=random_seed)
    HMM_Simulator.run(n_t=number_rounds)

    Belief_simulator = HMMBeliefForward(vec_init=INIT_DISTRIBUTION_STATIONARY, transition_matrix=TRANSITION, emission_matrix=EMISSION)
    Belief_simulator.run(HMM_Simulator.hist_context)

    n_contexts, n_regimes = EMISSION.shape
    Spectral_direct_estimator = HMMSpectralDirectEstimation(n_contexts = n_contexts, n_regimes = n_regimes, random_seed=random_seed)
    Spectral_direct_estimator.run(HMM_Simulator.hist_context)

    for index_, context_hmm_ in enumerate(HMM_Simulator.hist_context):
        print(index_)
        dt_filter_ = dt[dt['index_hmm'] == (context_hmm_ + 1)].reset_index(drop=True)
        dt_filter_ = dt_filter_.sample(1)
        if index_ == 0:
            dt_env = dt_filter_
        else:
            dt_env = pd.concat([dt_env, dt_filter_], axis=0).reset_index(drop=True)

    dump(HMM_Simulator, f"{PATH_MODELS}/random_seed_{random_seed}/hmm_simulator.pkl")
    dump(Belief_simulator, f"{PATH_MODELS}/random_seed_{random_seed}/belief_simulator.pkl")
    dump(Spectral_direct_estimator, f"{PATH_MODELS}/random_seed_{random_seed}/spectral_direct_estimator.pkl")

    dt_env['index_env'] = dt_env.index + 1
    dt_env = dt_env[['index_env', 'index_context', 'index_hmm'] + [x for x in dt_env if x not in ['index_env', 'index_context', 'index_hmm'] ]]
    dt_env.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_env.pkl")

    ####################################################################################
    ####################################################################################
    ############################ Simulation Actions and Rewards#########################
    ####################################################################################
    ####################################################################################
    for index_, action in enumerate(list_actions):
        dt_ = deepcopy(dt_env)
        dt_['ACTION'] = action
        if index_ == 0:
            dt_reward = dt_
        else:
            dt_reward = pd.concat([dt_reward, dt_], axis=0).reset_index(drop=True)

    dt_reward_dummy = transform_dt_dummy(dt_reward)
    dt_reward['expected_reward_rally'] = model_reward_rally.predict(dt_reward_dummy)
    dt_reward['expected_reward_down'] = model_reward_down.predict(dt_reward_dummy)

    for regime_ in ['rally', 'down']:
        dt_reward[f'realized_reward_{regime_}'] = \
            dt_reward.apply(lambda x:np.random.normal(x[f'expected_reward_{regime_}'], noise_std), axis=1)

    df_regime = pd.DataFrame.from_dict(data={'index_env': range(1, number_rounds + 1),
                                             'regime': list(HMM_Simulator.hist_regime)})
    dt_reward = dt_reward.merge(df_regime, how = 'left', on = 'index_env').reset_index(drop=True)

    dt_reward['expected_reward'] = dt_reward['expected_reward_rally']
    dt_reward['realized_reward'] = dt_reward['realized_reward_rally']
    index_down = dt_reward['regime'] == 1
    dt_reward.loc[index_down, 'expected_reward'] = dt_reward.loc[index_down, 'expected_reward_down']
    dt_reward.loc[index_down, 'realized_reward'] = dt_reward.loc[index_down, 'realized_reward_down']

    ####################################################################################
    ####################################################################################
    ######## Generate the best action and rewards for known underlying state###########
    ####################################################################################
    ####################################################################################
    dt_reward_regime_action = \
        dt_reward.sort_values(['index_env', 'expected_reward'], ascending = False).\
            drop_duplicates('index_env').sort_values('index_env').reset_index(drop=True)

    print(dt_reward_regime_action.groupby('regime')['ACTION'].value_counts())
    print(dt_reward_regime_action['ACTION'].value_counts())

    ####################################################################################
    ####################################################################################
    ######## Generate the best action and rewards for belief state###########
    ####################################################################################
    ####################################################################################
    df_belief = pd.DataFrame(Belief_simulator.hist_belief)
    df_belief.columns = ['prob_rally', 'prob_down']
    df_belief['index_env'] =  range(1, number_rounds + 1)
    dt_reward_belief = dt_reward.merge(df_belief, how = 'left', on = 'index_env').reset_index(drop=True)
    dt_reward_belief['believed_reward'] = \
        dt_reward_belief['expected_reward_rally'] * dt_reward_belief['prob_rally'] + \
        dt_reward_belief['expected_reward_down'] * dt_reward_belief['prob_down']

    dt_reward_belief_action = \
        dt_reward_belief.sort_values(['index_env', 'believed_reward'], ascending = False).\
            drop_duplicates('index_env').sort_values('index_env').reset_index(drop=True)
    dt_reward_belief_action = \
        dt_reward_belief_action[['index_env', 'ACTION', 'prob_rally', 'prob_down','believed_reward']]
    print(dt_reward_belief_action['ACTION'].value_counts())

    dt_reward_belief_action = dt_reward.merge(dt_reward_belief_action,
                                              how = 'inner', on =['index_env',  'ACTION']).sort_values('index_env').reset_index(drop=True)

    print(dt_reward_belief_action.groupby('regime')['ACTION'].value_counts())
    print(dt_reward_belief_action['ACTION'].value_counts())

    ####################################################################################
    ####################################################################################
    ######## Export the results###########
    ####################################################################################
    ####################################################################################
    dict_kpi_benchmark = {}
    dict_kpi_benchmark['index_env'] = np.array(range(1, number_rounds + 1))
    dict_kpi_benchmark['index_context'] = dt_env['index_context'].values
    dict_kpi_benchmark['index_hmm'] = dt_env['index_hmm'].values
    dict_kpi_benchmark['regime'] = np.array([int(x) for x in HMM_Simulator.hist_regime])

    dict_kpi_benchmark['optim_action_regime'] = dt_reward_regime_action['ACTION'].values
    dict_kpi_benchmark['optim_expected_reward_regime'] = dt_reward_regime_action['expected_reward'].values
    dict_kpi_benchmark['optim_realized_reward_regime'] = dt_reward_regime_action['realized_reward'].values

    dict_kpi_benchmark['prob_rally'] = dt_reward_belief_action['prob_rally'].values
    dict_kpi_benchmark['prob_down'] = dt_reward_belief_action['prob_down'].values

    dict_kpi_benchmark['optim_action_belief'] = dt_reward_belief_action['ACTION'].values
    dict_kpi_benchmark['optim_believed_reward_belief'] = dt_reward_belief_action['believed_reward'].values
    dict_kpi_benchmark['optim_expected_reward_belief'] = dt_reward_belief_action['expected_reward'].values
    dict_kpi_benchmark['optim_realized_reward_belief'] = dt_reward_belief_action['realized_reward'].values

    dump(dict_kpi_benchmark, f"{PATH_MODELS}/random_seed_{random_seed}/dict_kpi_rewards.pkl")
    dt_reward.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward.pq")
    dt_reward_dummy.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_dummy.pq")
    dt_reward_regime_action.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_regime_action.pq")
    dt_reward_belief_action.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_belief_action.pq")
    dt_reward_belief.to_parquet(f"{PATH_MODELS}/random_seed_{random_seed}/dt_reward_belief.pq")
