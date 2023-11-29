

import pandas as pd
import numpy as np 
import os
from scipy.stats import skew, kurtosis
import random

# Normalization    
def _vital_norm(df4):
    _df4 = df4
    for p in range(len(df4.columns)):
        if df4.columns[p] != '' and df4.columns[p] != 'stay_id' and df4.columns[p] != 'vital_time':
            if df4.columns[p] == 'heart_rate':
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 30) / (250-30)
            elif df4.columns[p] == 'mbp':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 20) / (250-20)
            elif df4.columns[p] == 'resp_rate':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 5) / (60-5)
            elif df4.columns[p] == 'sbp':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 30) / (250-30)
            elif df4.columns[p] == 'dbp':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 10) / (230 - 10)
            elif df4.columns[p] == 'spo2':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 30) / (100-30)        
            elif df4.columns[p] == 'temperature':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 34.5) / (38-34.5)   
            elif df4.columns[p] == 'EWS_SP02':
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 0.42) / 0.80 
            elif df4.columns[p] == 'EWS_TEMP':        
                df4.iloc[:,p] = (np.nan_to_num(df4.iloc[:,p]) - 0.04) / 0.13  
    return _df4

# Multi-resolution statistical feature generation
def _extract_multivew_stat_feat(df_new1, time, view):
    _column_name = df_new1.columns[3:]
    _mean = []
    _std = []
    _median = []
    _min = []
    _max = []
    for c_name in _column_name:    
        for n in range(view, time + 1, view):        
            
            _mean.append(df_new1[c_name][n-view:n].astype('float').mean())
            _std.append(df_new1[c_name][n-view:n].astype('float').std())
            _median.append(df_new1[c_name][n-view:n].astype('float').median())
            _min.append(df_new1[c_name][n-view:n].astype('float').min())
            _max.append(df_new1[c_name][n-view:n].astype('float').max())
        
    _stat_feat = np.reshape(np.concatenate([np.array(_mean), np.array(_std), np.array(_median), np.array(_min), np.array(_max)], axis = 0), (1,-1))
    return _stat_feat
    
def _extract_stat_feat(df_new1, time):
    
    _4stat_feat = _extract_multivew_stat_feat(df_new1, time, 4)
    _6stat_feat = _extract_multivew_stat_feat(df_new1, time, 6)
    if time == 12:
        _global_feat = _extract_multivew_stat_feat(df_new1, time, 12)
        _lg_stat_feat = np.concatenate([_4stat_feat, _6stat_feat, _global_feat], axis = 1)
    else:
        _12stat_feat = _extract_multivew_stat_feat(df_new1, time, 12)
        _global_feat = _extract_multivew_stat_feat(df_new1, time, 24)
        _lg_stat_feat = np.concatenate([_4stat_feat, _6stat_feat, _12stat_feat, _global_feat], axis = 1)
    return _lg_stat_feat

def _extract_multivew_cos_stat_feat(df_new1, time, view):

    _mean = []
    _std = []
    _min = []
    _max = []
    
    for n in range(view, time + 1, view):        
        
        _mean.append(df_new1[n-view:n].astype('float').mean())
        _std.append(df_new1[n-view:n].astype('float').std())
        _min.append(df_new1[n-view:n].astype('float').min())
        _max.append(df_new1[n-view:n].astype('float').max())
        
    _stat_feat = np.reshape(np.concatenate([np.array(_mean), np.array(_std), np.array(_min), np.array(_max)], axis = 0), (1,-1))
    return _stat_feat

# Cosien similarity feature generation
def _cosine_similarity(data, time, chan):

    from scipy import sparse
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = np.zeros((data.shape[0], chan, chan))
    for n in range(data.shape[0]):
        cos_sim[n,:,:] = cosine_similarity(sparse.csr_matrix(data[n,:,:].transpose()))
    cos_sim_tran = np.zeros((data.shape[0], time, time))
    for n in range(data.shape[0]):
        cos_sim_tran[n,:,:] = cosine_similarity(sparse.csr_matrix(data[n,:,:]))
    return cos_sim, cos_sim_tran