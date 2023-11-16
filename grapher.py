import matplotlib.pyplot as plt
import seaborn as sns

correct = 5
second = 1
third = 2
fifth = 1
miss = 1

# Labels for the pie chart
labels = ['Correct', 'MVP 2nd', 'MVP 3rd', 'MVP 5th', 'MVP not in top 5']

# Data for the pie chart
sizes = [correct, second, third, fifth, miss]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# Add a title
plt.title('MVP Predictions for Testing Data (2000-2003, 2023): ML Model', fontweight='bold')

plt.text(0, -1.25, '* In-Sample Testing (2004-2022) for ML model yielded 100% accuracy', ha='center', va='center', fontsize=10)


# Save the plot as a PDF file
plt.savefig('mvp_predictions_test_data.pdf', format='pdf')

plt.close() 

correct = 23
second = 6
third = 6 
fourth = 2
miss = 4

# Labels for the pie chart
labels = ['Correct', 'MVP 2nd', 'MVP 3rd', 'MVP 4th', 'MVP not in top 5']

# Data for the pie chart
sizes = [correct, second, third, fourth, miss]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Create a pie chart
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# Add a title
plt.title('MVP Predictions (2000-2022): Manual Model', fontweight='bold')

# Save the plot as a PDF file
plt.savefig('mvp_predictions_manual_model.pdf', format='pdf')

plt.close() 

stats = {
    'BA': 3.999444444,  
    'HR': 3.39748928,  
    'RBI': 2.20860146, 
    'OPS': 5.6818232,
    'WAR': 10.1,
    'SB': 0.71136828, 
    'ROBA': 5.8043232,
    'DEF': 0.18179077,
    'Win%': 6.37843602
}


# Sort the stats by importance in descending order
sorted_stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}

# Extract the sorted keys and values
features = list(reversed(list(sorted_stats.keys())))
importances = list(reversed(list(sorted_stats.values())))

# Define a list of colors for each bar
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

# Create a bar plot for feature importances with custom colors
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color=colors)
plt.title('Feature Importances for Manual MVP Model', fontweight='bold')
plt.xlabel('Assigned Weight', fontweight='bold')
plt.ylabel('Statistic', fontweight='bold')


# Save the plot as a PDF file
plt.savefig('feature_importances_manual_mvp_model.pdf', format='pdf')

# Close the plot (if you don't want to display it)
plt.close()




