import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

for x in range(2004, 2023):
    if x == 2020:
        continue
    # Load your dataframe (replace 'data.csv' with your actual data file)
    data = pd.read_csv(f'./training_data/full_season_data/{x}_composite.csv')
    
    # Define the categorical variables to be one-hot encoded
    categorical_vars = ['Tm', 'Last Name', 'First Name', 'Pos']
    
    # Encode categorical features using one-hot encoding
    data = pd.get_dummies(data, columns=categorical_vars)
    
    # Define columns to be dropped
    columns_to_drop = ['MVP']
    if 'Lg' in data:
        columns_to_drop.append('Lg')
    if 'Rk' in data:
        columns_to_drop.append('Rk')
    
    # Define features and target
    X = data.drop(columns=columns_to_drop)  # Remove unnecessary columns
    y = data['MVP']
    y.fillna(0, inplace=True)
    
    # Create and train XGBoost model
    model = XGBClassifier()
    model.fit(X, y)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Create a DataFrame to store feature importance scores and their corresponding column names
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    
    # Sort features by importance in descending order
    sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Print the top features
    print(f'{x} results: {sorted_features.head()}\n')
    
    
# War: 1 = 5 times, 
# OPS: 1 = 8 times, 
# HR: 1 = 1 time
# BA: 2 times
# 
