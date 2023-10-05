#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:34:36 2023

@author: Alex
"""

import pandas as pd
import joblib

# Step 1: Load the NL data model
al_model = joblib.load('al_xgb_model.pkl')

# Step 2: Load your 2018 CSV data into a DataFrame
# Replace '2018_data.csv' with the actual path to your 2018 data file
data_2018 = pd.read_csv('./training_data/full_season_data/2012_composite.csv')

# Step 3: Split the data into AL and NL DataFrames
al_data = data_2018[data_2018['Lg'] == 'AL']
#categorical_vars = ['Pos']

# Encode categorical features using one-hot encoding
#nl_data = pd.get_dummies(nl_data, columns=categorical_vars)

# Define columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos']

# Assuming 'MVP' is the target variable you want to predict
# You may also need to preprocess the data (e.g., drop unnecessary columns, handle missing values) as needed

# Step 4: Use the NL model to make predictions for the NL data
al_X = al_data.drop(columns=columns_to_drop)  # Remove the target variable
al_predictions = al_model.predict(al_X)

# Add the predictions to the NL DataFrame
al_data['MVP_Predicted'] = al_predictions

# Print or save the NL DataFrame with predictions
print("AL Data with Predicted MVP:")
print(al_data)


