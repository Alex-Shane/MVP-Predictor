import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./training_data/full_season_data/tester.csv')

# columns to be dropped
columns_to_drop = ['MVP', 'Lg', 'Rk', 'First Name', 'Last Name', 'Tm', 'Pos', 'GS', 'G', 'PA']

X = data.drop(columns=columns_to_drop)
Y = data['MVP']
Y.fillna(0,inplace=True)

# create and train XGBoost model
mvp_model = XGBClassifier(max_depth=6)
mvp_model.fit(X,Y)

# get feature importances
mvp_feature_importances = mvp_model.feature_importances_
mvp_feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': mvp_feature_importances})

# sort features by importance in descending order
mvp_sorted_features = mvp_feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top features
print("Total Results: ")
print(mvp_sorted_features)

# save model 
joblib.dump(mvp_model, 'mvp_model.pkl')

"""
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
plt.show()

# Close the plot (if you don't want to display it)
plt.close()"""

#joblib.dump(al_model, 'al_xgb_model.pkl')
#joblib.dump(nl_model, 'nl_xgb_model.pkl')

    
    
