# Source code for the In-Hospital Cardiac Arrest Prediction

This project contains the scripts associated to the manuscript "Explainable AI Warning Model Using Ensemble Approach for In-Hospital Cardiac Arrest Prediction: A Retrospective Cohort Study".  

If you wish to use this source code, please cite the "Journal of Medical Internet Research (JMIR)" journal below.

- Programming Language: Python
- Contact: Yun Kwan Kim (ykwin@korea.ac.kr)
- Kim YK, Koo JH, Lee SJ, Song HS, Lee M. Explainable AI Warning Model Using Ensemble Approach for In-Hospital Cardiac Arrest Prediction: A Retrospective Cohort Study. JMIR. 2023

# Step 1: Vital signs preprocessing and feature generation
The vital signs including heart rate, temperature, systolic blood pressure (BP), diastolic BP, mean BP, SPo2, and respiration rate were generated with statistical and cosine similarity-based features.

# Step 2: Training light gradient boosting machine (LGBM)
The statistical and cosine similarity-based features were aggregated and were trained using LGBM.

# ETC
Numpy & Pandas & Scikit  

  The Numpy package is freely available at https://pypi.org/project/numpy/.  
  
  The Pandas package is freely available at https://pypi.org/project/pandas/.  
    The scikit package is freely available at https://scikit-learn.org/stable/.
