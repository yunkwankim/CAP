# Source code for the In-Hospital Cardiac Arrest Prediction

This project contains the scrips associated to the manuscript "Explainable AI Warning Model Using Ensemble Approach for In-Hospital Cardiac Arrest Prediction: A Retrospective Cohort Study"

- Programming Language: Python
- Contact: Yun Kwan Kim (ykwin@korea.ac.kr)

# Step 1: Vital signs preprocessing and feature generation
The vital signs including heart rate, temperature, systolic blood pressure (BP), diastolic BP, mean BP, SPo2, and respiration rate were generated statistical and cosine similarity-based features.

# Step 2: Training light gradient boosting machine (LGBM)
The statistical and cosine similarity based feature were aggerated and were trained using LGBM.
