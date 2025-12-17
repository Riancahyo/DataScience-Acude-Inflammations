# Visualisasi Perbandingan Metrik Utama (Accuracy dan F1-Score)

models = comparison_df['Model']
accuracy_scores = comparison_df['Accuracy']
f1_scores = comparison_df['F1-Score']
training_times = comparison_df['Training Time (s)']

bar_width = 0.3
index = np.arange(len(models))

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Accuracy dan F1-Score
bar1 = ax1.bar(index - bar_width/2, accuracy_scores, bar_width, label='Accuracy', color='skyblue')
bar2 = ax1.bar(index + bar_width/2, f1_scores, bar_width, label='F1-Score', color='lightcoral')

ax1.set_xlabel('Model')
ax1.set_ylabel('Score (Accuracy/F1-Score)', color='black')
ax1.set_xticks(index)
ax1.set_xticklabels(models, rotation=0)
ax1.set_title('Perbandingan Performa Model (Accuracy dan F1-Score)')
ax1.legend(loc='upper left')
ax1.set_ylim(0.95, 1.05)

# Plot Training Time
ax2 = ax1.twinx()
ax2.plot(index, training_times, color='green', marker='o', linestyle='--', linewidth=2, label='Training Time (s)')
ax2.set_ylabel('Training Time (s)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()

