#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:34:36 2023

@author: Alex
"""

import pandas as pd
import joblib

def inSampleTesting(model):
    for x in range(2004,2023):
        if x == 2020:
            continue
        data = pd.read_csv(f'./training_data/full_season_data/{x}_composite.csv')
        al_data = data[data['Lg'] == 'AL']
        nl_data = data[data['Lg'] == 'NL']

        # Define columns to be dropped
        columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos']

        # Step 4: Prepare the data for predictions
        X_al = al_data.drop(columns=columns_to_drop)  # Remove the target variable
        X_nl = nl_data.drop(columns=columns_to_drop)

        # Step 5: Use the AL model to get predicted probabilities for the AL data
        y_al_pred_prob = model.predict_proba(X_al)[:, 1]
        y_nl_pred_prob = model.predict_proba(X_nl)[:, 1]

        # Step 6: Add predicted MVP probabilities to the DataFrame
        al_data['Predicted MVP Probability'] = y_al_pred_prob
        nl_data['Predicted MVP Probability'] = y_nl_pred_prob

        # Step 7: Sort the DataFrame by predicted MVP probabilities in descending order
        al_data_sorted = al_data.sort_values(by='Predicted MVP Probability', ascending=False)
        nl_data_sorted = nl_data.sort_values(by='Predicted MVP Probability', ascending=False)
        
        mvp_al = al_data_sorted[['First Name', 'Last Name', 'Predicted MVP Probability']].head(1)
        mvp_nl = nl_data_sorted[['First Name', 'Last Name', 'Predicted MVP Probability']].head(1)

        print(f'Model AL MVP in {x}: {mvp_al}')
        print(f'Model NL MVP in {x}: {mvp_nl}')



# Step 1: Load the AL model
model = joblib.load('mvp_model.pkl')

# Step 2: Load the testing data (replace '2023_composite.csv' with your testing data file)
data = pd.read_csv('./testing_data/2000_composite.csv')

# Step 3: Filter the data by league
al_data = data[data['Lg'] == 'AL']
nl_data = data[data['Lg'] == 'NL']

# Define columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos']

# Step 4: Prepare the data for predictions
X_al = al_data.drop(columns=columns_to_drop)  # Remove the target variable
X_nl = nl_data.drop(columns=columns_to_drop)

# Step 5: Use the AL model to get predicted probabilities for the AL data
y_al_pred_prob = model.predict_proba(X_al)[:, 1]
y_nl_pred_prob = model.predict_proba(X_nl)[:, 1]

# Step 6: Add predicted MVP probabilities to the DataFrame
al_data['Predicted MVP Probability'] = y_al_pred_prob
nl_data['Predicted MVP Probability'] = y_nl_pred_prob

# Step 7: Sort the DataFrame by predicted MVP probabilities in descending order
al_data_sorted = al_data.sort_values(by='Predicted MVP Probability', ascending=False)
nl_data_sorted = nl_data.sort_values(by='Predicted MVP Probability', ascending=False)

# Step 8: Get the top 5 candidates
top_5_mvp_candidates_al = al_data_sorted[['First Name', 'Last Name', 'Predicted MVP Probability']].head(5)
top_5_mvp_candidates_nl = nl_data_sorted[['First Name', 'Last Name', 'Predicted MVP Probability']].head(5)

# Step 9: Output the top 5 candidates
print("Top 5 MVP Candidates (AL):")
print(top_5_mvp_candidates_al)
print("Top 5 MVP Candidates (NL):")
print(top_5_mvp_candidates_nl)

#inSampleTesting(model)

# in sample testing (AL): 16/16
# in sample testing (NL): 17/17




