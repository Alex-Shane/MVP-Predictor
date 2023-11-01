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




