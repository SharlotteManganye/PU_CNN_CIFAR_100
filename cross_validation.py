import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import yaml
import pandas as pd
from datetime import datetime
import pytz
import numpy as np # Import numpy for std calculations

# Import custom modules
from utils import load_config, print_section
from train import train # Now expecting train to return a dict of metrics for the last/best epoch
from test import test # Assuming your test function is in test.py (optional, for final evaluation)

# Import model definitions (ensuring correct lowercase names as per your run.py)
from models import model_1, model_2, model_3, model_4, model_5
from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5

def get_full_dataset_with_targets(data_set_id, transform):
    """
    Loads the full dataset (combining train and test splits) and returns the dataset
    along with its targets, which are needed for StratifiedKFold.
    """
    if data_set_id == 1:  # CIFAR 10
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.tensor(train_dataset.targets + test_dataset.targets)
    elif data_set_id == 2:  # CIFAR 100
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.tensor(train_dataset.targets + test_dataset.targets)
    elif data_set_id == 3:  # MNIST
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.tensor(train_dataset.targets + test_dataset.targets)
    else:
        raise ValueError(f"Invalid data_set_id: {data_set_id}")
    return full_dataset, targets


def run_cross_validation(config_filename, base_results_dir='results', n_splits=5):
    """
    Runs k-fold cross-validation based on the provided configuration.
    """
    base_path = os.getcwd()
    config_path = os.path.join(base_path, "configs", config_filename)
    config = load_config(config_path)

    # Allocate config variables
    program_config = config["program"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    # Program variables
    seed = program_config["seed"]
    gen_rand_seed = program_config["gen_rand_seed"]
    gpu = program_config["gpu"]

    if gen_rand_seed:
        torch.manual_seed(datetime.now().microsecond % (2**32 - 1))
    else:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Model variables (using .get() for optional parameters that might not be in all configs)
    model_id = model_config["model_id"]
    number_channels = model_config["number_channels"]
    number_classes = model_config["number_classes"]
    out_channels = model_config.get("out_channels")
    image_height = model_config.get("image_height") # Ensure these are present in config or derive them
    image_width = model_config.get("image_width")   # Example: For MNIST 28x28, CIFAR 32x32
    fc_hidden_size = model_config.get("fc_hidden_size")
    fc_dropout_rate = model_config.get("fc_dropout_rate")
    num_layers = model_config.get("num_layers")

    # Training variables
    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]

    loss_func = nn.CrossEntropyLoss()

    # Define transforms (adjust this if your models require specific normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization here if your models expect it.
        # Example: transforms.Normalize((mean,), (std,)) for grayscale
        # Example: transforms.Normalize((mean_r, mean_g, mean_b), (std_r, std_g, std_b)) for RGB
    ])
    
    # Load the full dataset and its targets for StratifiedKFold
    full_dataset, targets = get_full_dataset_with_targets(data_set_id, transform)

    # Initialize StratifiedKFold for balanced splits across classes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_fold_metrics_raw = [] # To store the dictionaries returned by train()

    print_section(f"{n_splits}-Fold Cross-Validation")

    for fold, (train_indices, val_indices) in enumerate(skf.split(full_dataset, targets)):
        print_section(f"Fold {fold+1}/{n_splits}")

        # Create data subsets for this fold
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Create data loaders for this fold
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

        # Model instantiation for this fold (matching your run.py)
        if model_id == 1:
            model = model_1(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 2:
            model = model_2(
                number_channels, out_channels, image_height, image_width,
                fc_dropout_rate,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 3:
            model = model_3(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 4:
            model = model_4(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        elif model_id == 5:
            model = model_5(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        elif model_id == 6:
            model = baseline_model_1(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 7:
            model = baseline_model_2(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 8:
            model = baseline_model_3(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 9:
            model = baseline_model_4(
                in_channels=number_channels,
                out_channels=out_channels,
                image_height=image_height,
                image_width=image_width,
                fc_hidden_size=fc_hidden_size,
                number_classes=number_classes,
                fc_dropout_rate=fc_dropout_rate,
                num_layers=num_layers,
            )
        elif model_id == 10:
            model = baseline_model_5(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        else:
            raise ValueError(f"Invalid model ID: {model_id}")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model for this fold; expecting it to return the last/best epoch's metrics dictionary
        last_epoch_metrics = train(
            model,
            train_loader,
            optimizer,
            loss_func,
            epochs,
            device,
            val_loader=val_loader,
            config_filename=config_filename
        )
        
        # Add fold number to the metrics
        last_epoch_metrics['Fold'] = fold + 1
        all_fold_metrics_raw.append(last_epoch_metrics)

    print_section("Cross-Validation Summary")
    
    # Process collected raw metrics into a DataFrame for presentation and export
    df_metrics = pd.DataFrame(all_fold_metrics_raw)

    # Calculate averages
    avg_train_loss = df_metrics['Train_Loss'].mean()
    avg_val_loss = df_metrics['Val_Loss'].mean()
    avg_train_acc = df_metrics['Train_Accuracy'].mean()
    avg_val_acc = df_metrics['Val_Accuracy'].mean()

    # Calculate standard deviations
    std_train_loss = df_metrics['Train_Loss'].std()
    std_val_loss = df_metrics['Val_Loss'].std()
    std_train_acc = df_metrics['Train_Accuracy'].std()
    std_val_acc = df_metrics['Val_Accuracy'].std()

    # Collect results for export to Excel as per user's requested format
    fold_results_for_export = []
    for index, row in df_metrics.iterrows():
        fold_results_for_export.append(
            {
                "Fold": row["Fold"],
                "Train Loss": row["Train_Loss"],
                "Val Loss": row["Val_Loss"],
                "Train Accuracy": row["Train_Accuracy"],
                "Val Accuracy": row["Val_Accuracy"],
            }
        )

    fold_df = pd.DataFrame(fold_results_for_export)

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

    # Print the formatted table
    print(fold_df.to_string(index=False))

    # Export the table to a CSV file with dynamic naming
    sa_timezone = pytz.timezone('Africa/Johannesburg') 
    current_time_sast = datetime.now(sa_timezone)
    current_time_str = current_time_sast.strftime("%Y%m%d_%H%M%S")
    yaml_base = os.path.splitext(config_filename)[0]
    
    cv_export_filename = os.path.join(base_results_dir, 'cross_validation', f"crossv_{yaml_base}_results_{current_time_str}.csv")
    os.makedirs(os.path.dirname(cv_export_filename), exist_ok=True) # Ensure directory exists

    fold_df.to_csv(cv_export_filename, index=False)
    print(f"\nDetailed cross-validation results exported to {cv_export_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation for a model.")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="model_1.yaml", # Default config file
        type=str,
        help="Path to the YAML config file for cross-validation.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5).",
    )
    args = parser.parse_args()

    run_cross_validation(args.config_file, n_splits=args.n_splits)