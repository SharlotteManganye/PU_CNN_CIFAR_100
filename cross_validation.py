import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

from train import *


def cross_validate(
    model, dataset, num_folds, batch_size, epochs, learning_rate, device, seed
):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split the train_dataset into train and validation subsets using the indices from KFold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Reinitialize the model for each fold
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()

        # Train the model for the current fold
        fold_train_loss, fold_train_acc = train(
            model, train_loader, optimizer, loss_func, epochs, device
        )
        fold_val_loss, fold_val_acc = val(model, val_loader, loss_func, device)

        # Store the results
        fold_train_losses.append(fold_train_loss)
        fold_val_losses.append(fold_val_loss)
        fold_train_accuracies.append(fold_train_acc)
        fold_val_accuracies.append(fold_val_acc)

        # Print the training accuracy for the current fold
        print(f"  Fold {fold + 1} Train Accuracy: {fold_train_acc:.4f}")

    # Calculate averages and standard deviations across all folds
    avg_train_loss = np.mean(fold_train_losses)
    std_train_loss = np.std(fold_train_losses)
    avg_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)

    avg_train_acc = np.mean(fold_train_accuracies)
    std_train_acc = np.std(fold_train_accuracies)
    avg_val_acc = np.mean(fold_val_accuracies)
    std_val_acc = np.std(fold_val_accuracies)

    print(f"\n--- Cross-validation Results ---")
    print(
        f"Average Train Loss: {avg_train_loss:.4f}, Std Train Loss: {std_train_loss:.4f}"
    )
    print(
        f"Average Validation Loss: {avg_val_loss:.4f}, Std Validation Loss: {std_val_loss:.4f}"
    )
    print(
        f"Average Train Accuracy: {avg_train_acc:.4f}, Std Train Accuracy: {std_train_acc:.4f}"
    )
    print(
        f"Average Validation Accuracy: {avg_val_acc:.4f}, Std Validation Accuracy: {std_val_acc:.4f}"
    )

    # Collect results for export to Excel
    fold_results = []
    for i in range(num_folds):
        fold_results.append(
            {
                "Fold": i + 1,
                "Train Loss": fold_train_losses[i],
                "Val Loss": fold_val_losses[i],
                "Train Accuracy": fold_train_accuracies[i],
                "Val Accuracy": fold_val_accuracies[i],
            }
        )

    # Create DataFrame from the fold results
    fold_df = pd.DataFrame(fold_results)

    # Add averages and standard deviations as new rows at the end
    avg_std_results = {
        "Fold": "Average",
        "Train Loss": avg_train_loss,
        "Val Loss": avg_val_loss,
        "Train Accuracy": avg_train_acc,
        "Val Accuracy": avg_val_acc,
    }

    std_results = {
        "Fold": "Std Dev",
        "Train Loss": std_train_loss,
        "Val Loss": std_val_loss,
        "Train Accuracy": std_train_acc,
        "Val Accuracy": std_val_acc,
    }
    fold_df = pd.concat(
        [fold_df, pd.DataFrame([avg_std_results, std_results])], ignore_index=True
    )

    # Export the table to an Excel file
    fold_df.to_csv("cross_validation_results_lr004.csv", index=False)

    return (
        avg_train_loss,
        std_train_loss,
        avg_val_loss,
        std_val_loss,
        avg_train_acc,
        std_train_acc,
        avg_val_acc,
        std_val_acc,
    )
