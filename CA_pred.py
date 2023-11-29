

import pandas as pd
import numpy as np 
import numpy as np
import pandas as pd
import lightgbm
from Feature_proc_gen import _revnorm_extract_feat_subwise, _revca_extract_feat_subwise, _cosine_similarity, _vital_norm

time = 24

_df = 1

## Load EMR
path = '_path'

_clean_no_ews = pd.read_csv(path + str(time) + '_norm.csv')
df4 = pd.read_csv(path +str(time) + '_CA.csv')
df1 = pd.read_csv(path + str(time) + '_CA_start.csv')

tot = pd.concat([_clean_no_ews, df4], axis=0)

# Normalization
_tot = _vital_norm(tot)
_norm_db = _tot.iloc[:_clean_no_ews.shape[0], :]
_df4 = _tot.iloc[_clean_no_ews.shape[0]:, :]
 
# Feature generation
norm_ews, norm_stat_feat = _revnorm_extract_feat_subwise(_norm_db, time)
ca_ews, ca_ews_stat_feat = _revca_extract_feat_subwise(df1, _df4, time, int(_df))

__norm_ews = np.array(norm_ews)[:,:,3:].astype(float)
__norm_stat_feat = norm_stat_feat.astype(float)           
__ca_ews = np.array(ca_ews)[:,:,3:].astype(float)
__tot_ews = np.concatenate([__norm_ews, __ca_ews], axis = 0)

__ca_ews_stat_feat = ca_ews_stat_feat.astype(float)

__tot_stat_feat = np.concatenate([__norm_stat_feat, __ca_ews_stat_feat], axis = 0)

y_norm = np.repeat(0, repeats=np.array(__norm_ews).shape[0])
y_ca = np.repeat(1, repeats=np.array(__ca_ews).shape[0])

y = np.concatenate([y_norm, y_ca])

cos_sim, cos_sim_tran = _cosine_similarity(__tot_ews, time, __tot_ews.shape[2])
_cos_stat_tran = np.concatenate([np.mean(cos_sim_tran, axis = 1), np.mean(cos_sim_tran, axis = 2), np.std(cos_sim_tran, axis = 1), np.std(cos_sim_tran, axis = 2)], axis = 1)
_cos_stat = np.concatenate([np.mean(cos_sim, axis = 1), np.mean(cos_sim, axis = 2), np.std(cos_sim, axis = 1), np.std(cos_sim, axis = 2)], axis = 1)
_cos_feat = np.concatenate([_cos_stat, _cos_stat_tran], axis= 1)

X = np.concatenate([_cos_feat, __tot_stat_feat], axis = 1)

## Load Model
_cos_lgb = lightgbm.LGBMClassifier(max_depth = 1, learning_rate = 0.04, n_estimators = 500, scale_pos_weight = 200)
_cos_lgb.fit(X,y)

## Prediction
_y_hat = _cos_lgb.predict(X_valid)

# Probability
_y_hat_prob = _cos_lgb.predict_proba(X_valid)[:,1]




                