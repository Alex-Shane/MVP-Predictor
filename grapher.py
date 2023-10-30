import matplotlib.pyplot as plt
import seaborn as sns

# Define the evaluation metrics and their values
evaluation_metrics = ['PR AUC Score', 'F1 Score']
values = [0.393, 0.5]  # Replace with your actual metric values

# Choose a different color palette (e.g., "Set1")
custom_palette = sns.color_palette("Set1")

# Create a Seaborn bar plot with horizontal grid lines removed and a custom color palette
sns.set(style="whitegrid", font_scale=1.2, rc={"lines.linewidth": 2, 'grid.linestyle': '--', 'axes.grid': False})
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=evaluation_metrics, y=values, palette=custom_palette)

# Add labels and a title
plt.xlabel('Evaluation Metric', fontweight='bold')
plt.ylabel('Metric Value', fontweight='bold')
plt.title('Evaluation Metrics for MVP Predictor: ML Model', fontweight='bold')

# Display the values on top of the bars
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12)

# Save the plot as a PDF file
plt.savefig('evaluation_metrics_mvp_model.pdf', format='pdf')

# Close the plot (if you don't want to display it)
plt.close()

correct = 3
second = 3
third = 1 
fifth = 1
miss = 2

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




