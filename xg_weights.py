import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score
import joblib 

# Load your dataframe (replace 'data.csv' with your actual data file)
data = pd.read_csv('./training_data/full_season_data/tester.csv')

al_data = data[data['Lg'] == 'AL']
print(al_data['MVP'].dtype)
nl_data = data[data['Lg'] == 'NL']

# Define the categorical variables to be one-hot encoded
#categorical_vars = ['Pos']

# Encode categorical features using one-hot encoding
#al_data = pd.get_dummies(al_data, columns=categorical_vars)
#nl_data = pd.get_dummies(nl_data, columns=categorical_vars)

# Define columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos']

# Define features and target
X_al = al_data.drop(columns=columns_to_drop)  # Remove unnecessary columns
y_al = al_data['MVP']
y_al.fillna(0, inplace=True)

X_nl = nl_data.drop(columns=columns_to_drop)
y_nl = nl_data['MVP']
y_nl.fillna(0, inplace=True)

# Create and train XGBoost model
al_model = XGBClassifier()
al_model.fit(X_al, y_al)
nl_model = XGBClassifier()
nl_model.fit(X_nl, y_nl)

# Get feature importances
al_feature_importances = al_model.feature_importances_
nl_feature_importances = nl_model.feature_importances_ 

# Create a DataFrame to store feature importance scores and their corresponding column names
al_feature_importance_df = pd.DataFrame({'Feature': X_al.columns, 'Importance': al_feature_importances})
nl_feature_importance_df = pd.DataFrame({'Feature': X_nl.columns, 'Importance': nl_feature_importances})

# Sort features by importance in descending order
al_sorted_features = al_feature_importance_df.sort_values(by='Importance', ascending=False)
nl_sorted_features = nl_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top features
print("AL results: ")
print(al_sorted_features.head())
print ("NL results:")
print(nl_sorted_features.head())

joblib.dump(al_model, 'al_xgb_model.pkl')
joblib.dump(nl_model, 'nl_xgb_model.pkl')
    
    
