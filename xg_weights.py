import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

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

X = data.drop(columns=columns_to_drop)
Y = data['MVP']
Y.fillna(0,inplace=True)

# Create and train XGBoost model
al_model = XGBClassifier()
al_model.fit(X_al, y_al)
nl_model = XGBClassifier()
nl_model.fit(X_nl, y_nl)

mvp_model = XGBClassifier(max_depth=6)
mvp_model.fit(X,Y)

# Split the data into training and testing sets
"""X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
mvp_model = XGBClassifier(max_depth=6)
mvp_model.fit(X_train, Y_train)

# Predict probabilities for the test set
Y_pred_prob = mvp_model.predict_proba(X_test)[:, 1]  # Use the probability of being class 1 (MVP)

# Calculate Precision-Recall curve and PR AUC score
precision, recall, _ = precision_recall_curve(Y_test, Y_pred_prob)
pr_auc = auc(recall, precision)

threshold = 0.4  # You can adjust the threshold as needed
Y_pred = (Y_pred_prob > threshold).astype(int)
f1 = f1_score(Y_test, Y_pred)

print(f"PR AUC Score: {pr_auc}")
print(f'f1 Score: {f1}')"""


# Get feature importances
al_feature_importances = al_model.feature_importances_
nl_feature_importances = nl_model.feature_importances_ 
mvp_feature_importances = mvp_model.feature_importances_

# Create a DataFrame to store feature importance scores and their corresponding column names
al_feature_importance_df = pd.DataFrame({'Feature': X_al.columns, 'Importance': al_feature_importances})
nl_feature_importance_df = pd.DataFrame({'Feature': X_nl.columns, 'Importance': nl_feature_importances})
mvp_feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': mvp_feature_importances})

# Sort features by importance in descending order
al_sorted_features = al_feature_importance_df.sort_values(by='Importance', ascending=False)
nl_sorted_features = nl_feature_importance_df.sort_values(by='Importance', ascending=False)
mvp_sorted_features = mvp_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top features
print("AL results: ")
print(al_sorted_features.head())
print ("NL results:")
print(nl_sorted_features.head())
print("Total Results: ")
print(mvp_sorted_features)

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=mvp_sorted_features.head(12))  # Change '10' to the number of top features you want to display
plt.title('Top Feature Importances for MVP Model', fontweight='bold')
plt.xlabel('Weight', fontweight='bold')
plt.ylabel('Statistic', fontweight='bold')

# Adjust the plot's margins to prevent y-label from getting cut off
plt.subplots_adjust(left=0.2)  # Increase the left margin as needed

# Save the plot as a PDF file
plt.savefig('feature_importances_mvp_model.pdf', format='pdf')

# Close the plot (if you don't want to display it)
plt.close()

#joblib.dump(al_model, 'al_xgb_model.pkl')
#joblib.dump(nl_model, 'nl_xgb_model.pkl')
#joblib.dump(mvp_model, 'mvp_model.pkl')
    
    
