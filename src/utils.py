from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd
import xgboost as xgb
from numba import jit

INIT_DISTRIBUTION = np.array([0.5, 0.5])
INIT_DISTRIBUTION_STATIONARY = np.array([0.5714, 0.4286])
TRANSITION = np.array([[0.85, 0.2], [0.15, 0.8]])

EMISSION = np.array([[0.0665, 0.0089], [0.0956, 0.0202], [0.0661, 0.0361], [0.1634, 0.007],
                     [0.1242, 0.0373], [0.0628, 0.0098], [0.0967, 0.0228], [0.076 , 0.032],
                     [0.0407, 0.0093], [0.0664, 0.0265], [0.0048, 0.042], [0.0109, 0.0341],
                     [0.0025, 0.0913], [0.0002, 0.0746], [0.0058, 0.0636], [0.0697, 0.0855],
                     [0.0107, 0.1423],[0.031 , 0.048], [0.001 , 0.1101],[0.005 , 0.0986]])

LIST_FEATURE_DUMMY = \
    [
        'INTERCEPT', 'EDUCATION_zl_GRAD_SCHOOL', 'EDUCATION_zl_UNIVERSITY', 'EDUCATION_zl_HIGH_SCHOOL', 'EDUCATION_zl_OTHERS',
        'MARRIAGE_zl_MARRIED',  'MARRIAGE_zl_SINGLE', 'MARRIAGE_zl_OTHERS',
        'AGE_GRP_zl_A', 'AGE_GRP_zl_B', 'AGE_GRP_zl_C', 'AGE_GRP_zl_D', 'AGE_GRP_zl_E',
        'RISK_SCORE_zl_A', 'RISK_SCORE_zl_B', 'RISK_SCORE_zl_C', 'RISK_SCORE_zl_D', 'RISK_SCORE_zl_E',
        'REVENUE_GRP_zl_A', 'REVENUE_GRP_zl_B', 'REVENUE_GRP_zl_C', 'REVENUE_GRP_zl_D',
        'ACTION_zl_call', 'ACTION_zl_email',
        'REVENUE_GRP_ACTION_zl_A_call', 'REVENUE_GRP_ACTION_zl_A_email',
        'REVENUE_GRP_ACTION_zl_D_call', 'REVENUE_GRP_ACTION_zl_D_email',
        'RISK_SCORE_ACTION_zl_A_call', 'RISK_SCORE_ACTION_zl_A_email',
        'RISK_SCORE_ACTION_zl_B_call', 'RISK_SCORE_ACTION_zl_B_email',
        'RISK_SCORE_ACTION_zl_D_call', 'RISK_SCORE_ACTION_zl_D_email',
        'RISK_SCORE_ACTION_zl_E_call', 'RISK_SCORE_ACTION_zl_E_email'
    ]

LIST_FEATURE_DUMMY_INVARIANT = \
    [
        'INTERCEPT', 'PROB_RALLY', 'EDUCATION_zl_GRAD_SCHOOL', 'EDUCATION_zl_UNIVERSITY', 'EDUCATION_zl_HIGH_SCHOOL', 'EDUCATION_zl_OTHERS',
        'MARRIAGE_zl_MARRIED',  'MARRIAGE_zl_SINGLE', 'MARRIAGE_zl_OTHERS',
        'AGE_GRP_zl_A', 'AGE_GRP_zl_B', 'AGE_GRP_zl_C', 'AGE_GRP_zl_D', 'AGE_GRP_zl_E',
        'RISK_SCORE_zl_A', 'RISK_SCORE_zl_B', 'RISK_SCORE_zl_C', 'RISK_SCORE_zl_D', 'RISK_SCORE_zl_E',
        'REVENUE_GRP_zl_A', 'REVENUE_GRP_zl_B', 'REVENUE_GRP_zl_C', 'REVENUE_GRP_zl_D'
    ]

LIST_FEATURE_DUMMY_VARIANT = [x for x in LIST_FEATURE_DUMMY if x not in LIST_FEATURE_DUMMY_INVARIANT]
LIST_FEATURE_DUMMY_CB = LIST_FEATURE_DUMMY_INVARIANT + [f'{x}_rally' for x in LIST_FEATURE_DUMMY_VARIANT] + \
                        [f'{x}_down' for x in LIST_FEATURE_DUMMY_VARIANT]

def combine_dt_dummy_belief(dt, belief):
    r, c = belief.shape
    r_dt, _ = dt.shape
    if r!= r_dt:
        dt_belief = pd.DataFrame(belief)
        dt_belief.columns = ['PROB_RALLY', 'PROB_DOWN']
        dt_belief['index_env'] = list(range(1, r+1))
        dt = dt.merge(dt_belief, how='left', on='index_env').reset_index(drop=True)
    else:
        dt = deepcopy(dt)
        dt['PROB_RALLY'] = belief[:, 0]

    dt_invar = dt[LIST_FEATURE_DUMMY_INVARIANT].reset_index(drop=True)
    dt_var_rally = dt[LIST_FEATURE_DUMMY_VARIANT].reset_index(drop=True)
    if r != r_dt:
        for var_ in LIST_FEATURE_DUMMY_VARIANT:
            dt_var_rally[var_] = dt_var_rally[var_] * dt['PROB_RALLY']
    else:
        for var_ in LIST_FEATURE_DUMMY_VARIANT:
            dt_var_rally[var_] *= belief[:, 0]

    dt_var_rally.columns = [f'{x}_rally' for x in dt_var_rally]

    dt_var_down = dt[LIST_FEATURE_DUMMY_VARIANT].reset_index(drop=True)
    if r != r_dt:
        for var_ in LIST_FEATURE_DUMMY_VARIANT:
            dt_var_down[var_] = dt_var_down[var_] * dt['PROB_DOWN']
    else:
        for var_ in LIST_FEATURE_DUMMY_VARIANT:
            dt_var_down[var_] *= belief[:, 1]

    dt_var_down.columns = [f'{x}_down' for x in dt_var_down]
    dt_final = pd.concat([dt_invar, dt_var_rally, dt_var_down], axis=1).reset_index(drop=True)
    return dt_final[LIST_FEATURE_DUMMY_CB]

def transform_dt_dummy(dt):
    dt = deepcopy(dt)
    dt['INTERCEPT'] = 1
    dt['REVENUE_GRP_ACTION'] = dt['REVENUE_GRP'] + '_' + dt['ACTION']
    dt['RISK_SCORE_ACTION'] = dt['RISK_SCORE'] + '_' + dt['ACTION']

    list_feature = [x for x in dt if 'index' not in x]
    df_dummy = pd.get_dummies(dt[list_feature], prefix_sep='_zl_')
    df_dummy = df_dummy * 1

    output = \
        pd.concat([dt[[x for x in dt if 'index' in x] + ['ACTION']], df_dummy[LIST_FEATURE_DUMMY]], axis= 1).reset_index(drop=True)
    return output

@jit
def compute_belief_init_numba(context: int,
                              emission_matrix: np.ndarray,
                              current_belief: np.ndarray):
    prob = emission_matrix[context, :] * current_belief
    return prob / prob.sum()


@jit
def compute_belief_numba(context: int,
                         transition_matrix: np.ndarray,
                         emission_matrix: np.ndarray,
                         current_belief: np.ndarray):
    prob = emission_matrix[context, :] * (transition_matrix @ current_belief)
    return prob / prob.sum()


@jit
def update_belief_numba(transition_matrix: np.ndarray,
                        emission_matrix: np.ndarray,
                        vec_init: np.ndarray,
                        list_contexts: np.ndarray):
    T = len(list_contexts)
    H = vec_init.shape[0]
    hist_belief = np.empty((T, H), dtype=np.float64)

    current_belief = vec_init.copy()

    for t in range(T):
        context = list_contexts[t]
        if t == 0:
            current_belief = compute_belief_init_numba(context, emission_matrix, current_belief)
        else:
            current_belief = compute_belief_numba(context, transition_matrix, emission_matrix, current_belief)

        hist_belief[t, :] = current_belief

    return hist_belief

def load_xgb(MODEL_PATH, FEATURES_PATH, CORES=5):
    models = xgb.Booster({"nthread":CORES})
    models.load_model(MODEL_PATH)
    features = list(np.load(FEATURES_PATH, allow_pickle = True))
    models.feature_names = features
    return models

def predict_xgb(DATA, MODEL, PREFIX_SEP = "_zl_"):
    dt_dummy = pd.get_dummies(DATA, prefix_sep=PREFIX_SEP)
    features = MODEL.feature_names

    features_ = [x for x in features if x not in dt_dummy.columns]

    for var_ in features_:
        dt_dummy[var_] = 0

    dt_dummy = dt_dummy[features]

    xgb_data = xgb.DMatrix(data=dt_dummy.values,feature_names=features)

    return MODEL.predict(xgb_data)

def format_lmd(lmd_, sci_low=Decimal("1e-6")):
    d = Decimal(str(lmd_))

    if d == 0:
        return "0"

    def clean_number(s):
        return s.rstrip("0").rstrip(".").replace(".", "p")

    # Special case: very small values, e.g. 1e-17 -> 1em17
    if abs(d) < sci_low:
        sci = f"{d.normalize():E}"
        base, power = sci.split("E")

        base = clean_number(base)
        power = int(power)

        if power < 0:
            return f"{base}em{abs(power)}"
        return f"{base}e{power}"

    # Original backward-compatible behavior
    s = "{:f}".format(lmd_).rstrip("0").rstrip(".")

    if lmd_ < 1:
        return "0" + s[2:]

    return s.replace(".", "p")