import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the base directory and number of folds
base_dir = 'results/model_1'
num_folds = 2

# Initialize a list to store DataFrames from each fold
fold_data = []

# Read each fold's CSV and store the relevant columns
for i in range(1, num_folds + 1):
    file_path = os.path.join(base_dir, f'fold_{i}', f'metrics_fold_{i}.csv')
    df = pd.read_csv(file_path)
    fold_data.append(df[['Epoch', 'Train_Accuracy', 'Val_Accuracy']])

# Concatenate all data and compute mean accuracy per epoch
combined_df = pd.concat(fold_data)
mean_accuracies = combined_df.groupby('Epoch').mean().reset_index()

# Plotting the mean training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(mean_accuracies['Epoch'], mean_accuracies['Train_Accuracy'], label='Mean Train Accuracy', marker='o')
plt.plot(mean_accuracies['Epoch'], mean_accuracies['Val_Accuracy'], label='Mean Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Mean Train and Validation Accuracy per Epoch Across 5 Folds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mean_accuracy_plot.png')
plt.show()

