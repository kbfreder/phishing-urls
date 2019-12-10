'''This is the version for the src/models/ folder'''

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump

# GENERATE DICTIONARY

def generate_dict_pd(df):
    '''Return a look-up dictionary based on BAN and Category
        providing average amount (AMOUNT_USD_ROUNDED) and average number of records
        *per accounting period* (ACCT_PERIOD). Also supplies standard deviation.

        Stats (avg, std) are calculated from BAN-ACF groups, but dictionary is intended 
        to be applied to BAN-NCF group for featuring engineering.

        Dictionary `key: value` output is:
        (BAN, CAT): ['AVG_PD_TOTAL', 'STD_PD_TOTAL', 'AVG_NUM_ENTR', 'STD_NUM_ENTR', 'AVG_PCT_OVR', 'NUM_PDS']
            - 'AVG_PCT_OVR': average percent of BAN-CAT records overridden per period
            - 'NUM_PDS': number of periods for which there is data for BAN-CAT in df
        
        df: train data
    '''
    # agg over BAN, ACCT PD, ACF
    grp = df.groupby(['BAN', 'ACCT_PERIOD', 'ACCOUNTING_CAT_FINAL']).agg({
        'AMOUNT_USD_ROUNDED': [np.mean, np.std, np.sum, 'count'],
        'CALC_OVR_IND': np.mean
        })
    grp.columns = ['MEAN_USD', 'STD_USD', 'SUM_USD', 'NUM_ENTRIES', 'MEAN_OVR_PCT']
    grp.reset_index(inplace=True, drop=False)

    # agg over BAN, ACF
    meta_grp = grp.groupby(['BAN', 'ACCOUNTING_CAT_FINAL']).agg({
        'SUM_USD': [np.mean, np.std],
        'NUM_ENTRIES': [np.mean, np.std],
        'MEAN_OVR_PCT': [np.mean, 'count']
        })
    meta_grp.columns = ['AVG_PD_TOTAL', 'STD_PD_TOTAL', 'AVG_NUM_ENTR', 'STD_NUM_ENTR', 'AVG_PCT_OVR', 'NUM_PDS']
    meta_grp.reset_index(inplace=True, drop=False)

    # create look-up dictionary
    for_dict = meta_grp.set_index(keys=['BAN', 'ACCOUNTING_CAT_FINAL'])
    for_dict.fillna(0, inplace=True)
    ban_cat_pd_dict = for_dict.to_dict(orient='index')
    return ban_cat_pd_dict


def generate_dict_record(df):
    '''Return a look-up dictionary based on BAN and Category
        providing average amount (AMOUNT_USD_ROUNDED) *per BAN-CAT record*.
        Also supplies standard deviation, min, max, and number of records.

        Stats (avg, std) are calculated from BAN-ACF groups, but dictionary is intended 
        to be applied to BAN-NCF group for featuring engineering.

        Dictionary `key: value` output is:
        (BAN, CAT): ['AVG_USD', 'STD_USD', 'NUM_ENTRIES', 'MAX', 'MIN', 'AVG_PCT_OVR']
            - 'AVG_PCT_OVR': average percent of BAN-CAT records overridden per period
            - 'NUM_ENTRIES': number of records for which there is data for BAN-CAT in df
        
        df: train data
    '''

    grp = df.groupby(['BAN', 'ACCOUNTING_CAT_FINAL']).agg({'AMOUNT_USD_ROUNDED':[np.mean,  np.std, 'count', max, min],
                                                     'CALC_OVR_IND': np.mean})
    grp.columns = ['AVG_USD', 'STD_USD', 'NUM_ENTRIES', 'MAX', 'MIN', 'AVG_PCT_OVR']
    grp.reset_index(inplace=True, drop=False)
    grp.set_index(keys=['BAN', 'ACCOUNTING_CAT_FINAL'], inplace=True)
    grp = grp.fillna(0)
    
    ban_cat_amt_dict = grp.to_dict(orient='index')
    return ban_cat_amt_dict



#-----------------------------------
# CALCULATE Z-SCORES

# helper functions
def z_score(val, avg, std):
    if std == 0:
        return 0
    else:
        return np.abs((val - avg) / std)


def _row_zscore_pd(row, target_col, ban_dict):
    '''Calculate Z-score for total AMT_USD over Accounting Period for BAN & Category
        Z-score calculated relative to BAN-ACCOUNTING_CAT_FINAL average & std dev
        Category determined by target_col (NEW_CAT_FINAL or ACCOUNTING_CAT_FINAL)
    Notes: 
        - Function designed to return Z-score for every entry/row, even though it is applied to total for BAN-Category
        - Thus, data must already have 'AMT_BAN_NCF_PD' field created
    '''
    exp_vals = ban_dict.get((row['BAN'], row[target_col]), {'AVG_PD_TOTAL': 0, 'STD_PD_TOTAL': 0})
    avg = exp_vals['AVG_PD_TOTAL']
    std = exp_vals['STD_PD_TOTAL']
    val = row['AMT_BAN_NCF_PD']
    return z_score(val, avg, std)    


def _row_zscore_num_pd(row, target_col, ban_dict):
    '''Calculate Z-score for number of entries over Accounting Period for BAN & Category.
        Z-score calculated relative to BAN-ACCOUNTING_CAT_FINAL average & std dev
        Category determined by target_col (NEW_CAT_FINAL or ACCOUNTING_CAT_FINAL)
    Notes: 
        - Function designed to return Z-score for every entry/row, even though it is applied to total for BAN-Category
        - Thus, data must already have 'NUM_BAN_NCF_PD' field created
    '''
    exp_vals = ban_dict.get((row['BAN'], row[target_col]), {'AVG_NUM_ENTR': 0, 'STD_NUM_ENTR': 0})
    avg = exp_vals['AVG_NUM_ENTR']
    std = exp_vals['STD_NUM_ENTR']
    val = row['NUM_BAN_NCF_PD']
    return z_score(val, avg, std) 


def _row_zscore(row, ban_dict):
    '''Return z-score of AMOUNT_USD_ROUNDED for BAN-NCF vs average & standard deviation for BAN-ACF
    '''
    exp_vals = ban_dict.get((row['BAN'], row['NEW_CAT_FINAL']), {'AVG_USD': 0, 'STD_USD': 0})
    avg = exp_vals['AVG_USD']
    std = exp_vals['STD_USD']
    val = row['AMOUNT_USD_ROUNDED']
    return z_score(val, avg, std)


def calc_z_scores_pipeline(df, ban_dict_pd, ban_dict_record):
    Z_SCORE_AMT_PD = df.apply(lambda row: _row_zscore_pd(row, 'NEW_CAT_FINAL', ban_dict_pd), axis=1)
    Z_SCORE_NUM_PD = df.apply(lambda row: _row_zscore_num_pd(row, 'NEW_CAT_FINAL', ban_dict_pd), axis=1)
    Z_SCORE_AMT = df.apply(lambda row: _row_zscore(row, ban_dict_record), axis=1)

    result = np.transpose(np.array([Z_SCORE_AMT_PD, Z_SCORE_NUM_PD, Z_SCORE_AMT]))
    return result


#------------------------------------
# WEIGHT-OF-EVIDENCE

# settings
np.seterr(divide='ignore')

def _calc_row_proba(row):
    n = np.sum(row)
    vals = []
    for num_i in row:
        y_i = num_i
        proba = y_i / n
        vals.append(proba)
    
    return vals


def target_encode_pipeline(df):
    '''Calculate probability of ACCOUNTING-CAT-FINAL categories for BAN-NCF combo.
        df: train data. must already have 'BAN_NCF' field created.
        Returns a dictionary, with BAN-NCF as key; value = list of proba's
    '''
    ct = pd.crosstab(df['BAN_NCF'], df['ACCOUNTING_CAT_FINAL'])
    ct_woe_data = ct.apply(_calc_row_proba, axis=1)
    return ct_woe_data.to_dict()
