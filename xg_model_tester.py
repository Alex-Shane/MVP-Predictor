#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:34:36 2023

@author: Alex
"""

import pandas as pd
import joblib

# Step 1: Load the AL model
al_model = joblib.load('nl_xgb_model.pkl')

# Step 2: Load the testing data (replace '2023_composite.csv' with your testing data file)
al_data = pd.read_csv('./testing_data/2023_composite.csv')

# Step 3: Filter the data for AL only
al_data = al_data[al_data['Lg'] == 'NL']

# Define columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos']

# Step 4: Prepare the data for predictions
X_al = al_data.drop(columns=columns_to_drop)  # Remove the target variable

# Step 5: Use the AL model to get predicted probabilities for the AL data
y_al_pred_prob = al_model.predict_proba(X_al)[:, 1]

# Step 6: Add predicted MVP probabilities to the DataFrame
al_data['Predicted MVP Probability'] = y_al_pred_prob

# Step 7: Sort the DataFrame by predicted MVP probabilities in descending order
al_data_sorted = al_data.sort_values(by='Predicted MVP Probability', ascending=False)

# Step 8: Get the top 5 candidates
top_5_mvp_candidates = al_data_sorted[['First Name', 'Last Name', 'Predicted MVP Probability']].head(5)

# Step 9: Output the top 5 candidates
print("Top 5 MVP Candidates:")
print(top_5_mvp_candidates)


"""import pandas as pd
import joblib

# Step 1: Load the AL model
al_model = joblib.load('nl_xgb_model.pkl')

# Step 2: Load the testing data (replace '2023_composite.csv' with your testing data file)
testing_data = pd.read_csv('./testing_data/2023_composite.csv')

# Step 3: Filter the data for AL only
al_data = testing_data[testing_data['Lg'] == 'NL']

# Define columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos', 'G', 'GS']

# Step 4: Prepare the data for predictions
X_al = al_data.drop(columns=columns_to_drop)  # Remove the target variable

# Step 5: Use the AL model to make predictions for the AL data
y_al_pred = al_model.predict(X_al)

# Step 6: Create a DataFrame with First Name, Last Name, and Predicted MVP Values
predicted_mvp_df = pd.DataFrame({
    'First Name': al_data['First Name'],
    'Last Name': al_data['Last Name'],
    'Predicted MVP Value': y_al_pred
})

# Step 7: Sort the DataFrame by predicted MVP value in descending order
predicted_mvp_df = predicted_mvp_df.sort_values(by='Predicted MVP Value', ascending=False)

# Step 8: Get the top 5 players with the highest predicted MVP values
top_5_mvp_candidates = predicted_mvp_df.head(5)

# Print or save the top 5 MVP candidates
print("Top 5 MVP Candidates:")
print(top_5_mvp_candidates)"""




