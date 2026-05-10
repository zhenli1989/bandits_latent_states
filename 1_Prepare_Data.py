# coding: utf-8
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load

from src.model import AppearEstimator
from src.utils import LIST_FEATURE_DUMMY, predict_xgb, load_xgb

####################################################################################
####################################################################################
############################# AREA OF INPUT PARAMETERS #############################
####################################################################################
####################################################################################

##### Environment Parameters

PATH = "." # Root directory, should be the same path this "README.md" file locates
PATH_DATA = f"{PATH}/data" # Path for data
PATH_MODELS = f"{PATH}/models" # Path for models

retrain_model = True # Default False. If retrain_model is True, it will retrain the PD model. If False, it will load the pretrained model
random_seed = 1989 # Random seed, used when retrain_model is True. To reproduce the PD model, set as 1989

if os.path.isdir(f"{PATH_MODELS}/random_seed_{random_seed}"):
    pass
else:
    os.makedirs(f"{PATH_MODELS}/random_seed_{random_seed}")

####################################################################################
####################################################################################
############# Load the raw data and pretrained model ###############################
####################################################################################
####################################################################################
##### Set random seed
np.random.seed(random_seed)

##### Load raw data ######
dt_raw = pd.read_excel(f"{PATH_DATA}/default of credit card clients.xls",  skiprows=[0]).drop("SEX", axis=1)

##### Load pretrained PD model ######
if retrain_model is False:
    xgb_pd = load_xgb(f"{PATH_MODELS}/models_pd.model", f"{PATH_MODELS}/models_pd_features.npy")
    model_reward_rally = load(f"{PATH_MODELS}/model_reward_rally.pkl")
    model_reward_down = load(f"{PATH_MODELS}/model_reward_down.pkl")

####################################################################################
####################################################################################
############################## Prepare the raw dataset #############################
####################################################################################
####################################################################################

##### Mapping features
dict_mapping_edu = {0:"OTHERS", 1:"GRAD_SCHOOL", 2:"UNIVERSITY", 3:"HIGH_SCHOOL",
                    4:"OTHERS", 5:"OTHERS", 6:"OTHERS"}

dict_mapping_marriage = {0: "OTHERS", 1:"MARRIED", 2:"SINGLE", 3:"OTHERS"}

dt_raw["EDUCATION"] = dt_raw["EDUCATION"] .apply(lambda x:dict_mapping_edu[x])
dt_raw["MARRIAGE"] = dt_raw["MARRIAGE"] .apply(lambda x:dict_mapping_marriage[x])

##### Simulate PD MODEL and adjustment
var_target = "default payment next month"
var_IDs = ["ID"]
var_model = [x for x in dt_raw.columns if x not in var_IDs + [var_target]]

if retrain_model is True:
    xgb_params = {"eta":0.01, "seed":random_seed, "subsample":0.8, "objective":"reg:logistic",
                  "booster":"gbtree", "nthread":8, "max_depth":3, "min_child_weight":10, "colsample_bytree":1}
    dt_raw_dummy = pd.get_dummies(dt_raw[var_model], prefix_sep="_zl_")
    xgb_data = xgb.DMatrix(data = dt_raw_dummy.values, label=dt_raw[var_target].values,
                           feature_names=list(dt_raw_dummy.columns.values))
    xgb_cv_pd = xgb.cv(params=xgb_params, dtrain=xgb_data, num_boost_round=2000, verbose_eval=1, early_stopping_rounds=100,
                       metrics="logloss")
    best_iter = xgb_cv_pd["test-logloss-mean"].argmin()
    xgb_pd = xgb.train(params=xgb_params, dtrain=xgb_data, num_boost_round=best_iter)
    dump(dt_raw, f"{PATH_MODELS}/dt_raw.pkl")
    dump(xgb_pd, f"{PATH_MODELS}/models_pd.pkl")
    xgb_pd.save_model(f"{PATH_MODELS}/models_pd.model")
    np.save(f"{PATH_MODELS}/models_pd_features.npy", np.array(dt_raw_dummy.columns.values))

dt_raw["PD"] = predict_xgb(dt_raw, xgb_pd) # Rescale to make the PD more close to loan request

##### Define the requested amount

dt_raw["REVENUE"] = np.minimum(dt_raw["LIMIT_BAL"]*0.2, 100000)

var_keep = ["EDUCATION", "MARRIAGE", "REVENUE", "AGE", "PD"]


dt = dt_raw[var_IDs + var_keep].reset_index(drop = True)

##### Get Risk Score and Set Reference Rate
dict_mapping_risk_score = {"A":[-np.inf, 0.0864], "B":[0.0864, 0.1248], "C":[0.1248, 0.1748],
                           "D":[0.1748, 0.3136], "E":[0.3136, np.inf]}
# A is Level 1 to E is level 5 as in the paper

for key_ in dict_mapping_risk_score.keys():
    index_risk_ = (dt["PD"] > dict_mapping_risk_score[key_][0]) & (dt["PD"] <= dict_mapping_risk_score[key_][1])
    dt.loc[index_risk_, "RISK_SCORE"] = key_

dict_mapping_revenue = {"A":[54000, np.inf], "B":[36000, 54000], "C":[10000, 36000], "D":[-np.inf, 10000]}
# D is Level 1 to A is level 4 as in the paper

for key_ in dict_mapping_revenue.keys():
    index_revenue_ = (dt["REVENUE"] > dict_mapping_revenue[key_][0]) & (dt["REVENUE"] <= dict_mapping_revenue[key_][1])
    dt.loc[index_revenue_, "REVENUE_GRP"] = key_

dict_mapping_age = {"A":[-np.inf, 27], "B":[27, 31], "C":[31, 37], "D":[37, 43], "E":[43, np.inf]}
for key_ in dict_mapping_age.keys():
    index_age_ = (dt["AGE"] > dict_mapping_age[key_][0]) & (dt["AGE"] <= dict_mapping_age[key_][1])
    dt.loc[index_age_, "AGE_GRP"] = key_

##### Create clusters for Amount and Age
dict_map_class = dict(zip(['A', 'B', 'C', 'D', 'E'], [1, 2, 3, 4, 5]))
index_hmm = dt[['RISK_SCORE', 'REVENUE_GRP']].drop_duplicates().reset_index(drop=True)
index_hmm['index_sort'] = index_hmm['RISK_SCORE'].apply(lambda x:dict_map_class[x]) + index_hmm['REVENUE_GRP'].apply(lambda x:dict_map_class[x])
index_hmm = index_hmm.sort_values('index_sort').drop('index_sort', axis=1).reset_index(drop=True)
index_hmm['index_hmm'] = index_hmm.index + 1

##### Get the context index - Mapping context to index
var_context = ["RISK_SCORE", "EDUCATION", "MARRIAGE", "REVENUE_GRP", "AGE_GRP"]
context_list = dt[var_context].drop_duplicates().reset_index(drop = True)
context_list["index_context"] = context_list.index.values + 1
var_drop = ['PD', 'ID', 'REVENUE', 'AGE']
dt = dt.drop(var_drop, axis=1).\
    merge(context_list, how = "left", on = var_context).\
    merge(index_hmm, how = 'left', on = ['RISK_SCORE', 'REVENUE_GRP']).reset_index(drop = True)
dt = dt[['index_context', 'index_hmm'] + [x for x in dt if x not in ['index_context', 'index_hmm']]]

dt.to_parquet(f"{PATH_MODELS}/dt_ref.pq")
####################################################################################
####################################################################################
############################## Define the Reward Model #############################
####################################################################################
####################################################################################
### Model for Economic Rally
if retrain_model is True:
    coef_model_rally = dict()
    # list_coef_rally = [0.15, 0.1, 0.05, 0, 0, 0.1, 0, 0, 0, 0.025, 0.05, 0.025, 0,
    #                    0.2, 0.15, 0.1, 0.05, 0, 0.1, 0.075, 0.05, 0,
    #                    -0.2, 0.025, -0.2, 0.025,
    #                    0.2, -0.1, -0.2, 0.05, -0.1, 0.025, 0.3, -0.2, 0.4, -0.25]
    list_coef_rally = [0.05 , 0.033, 0.017, 0, 0, 0.033, 0, 0, 0, 0.008, 0.017, 0.008,
                       0, 0.067, 0.05, 0.033, 0.017, 0, 0.033, 0.025, 0.017, 0,
                       -0.2, 0.025, -0.2, 0.025,
                       0.2, -0.1, -0.2, 0.05, -0.1, 0.025, 0.3, -0.2, 0.4, -0.25]

    for index_, var_ in enumerate(LIST_FEATURE_DUMMY):
        coef_model_rally[var_] = list_coef_rally[index_]

    model_reward_rally = AppearEstimator(coef_model_rally)

    coef_model_down = dict()
    # list_coef_down = [0.05, 0.1, 0.05, 0, 0, 0.1, 0, 0,0, 0.025, 0.05, 0.025, 0,
    #                   0.2, 0.15, 0.1, 0.05, 0, 0.1, 0.075, 0.05, 0,
    #                   0.15, -0.1, 0.1, 0.05,
    #                   -0.2, 0.1, 0.25, 0.15, 0.2, 0.1, -0.1, 0.1, -0.3, 0.05]
    list_coef_down = [0.017 , 0.033, 0.017, 0, 0, 0.033, 0, 0, 0, 0.008, 0.017, 0.008,
                      0, 0.067, 0.05, 0.033, 0.017, 0, 0.033, 0.025, 0.017, 0,
                      0.15, -0.1, 0.1, 0.05,
                      -0.2, 0.1, 0.25, 0.15, 0.2, 0.1, -0.1, 0.1, -0.3, 0.05]
    for index_, var_ in enumerate(LIST_FEATURE_DUMMY):
        coef_model_down[var_] = list_coef_down[index_]

    model_reward_down = AppearEstimator(coef_model_down)

    dump(model_reward_rally, f"{PATH_MODELS}/model_reward_rally.pkl")
    dump(model_reward_down, f"{PATH_MODELS}/model_reward_down.pkl")
