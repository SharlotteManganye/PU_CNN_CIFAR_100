# cross_validation (3).py

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
import numpy as np

# Import custom modules
from utils import load_config, print_section
from train import train # Make sure this imports the modified train function
from test import test # Assuming your test function is in test.py (optional, for final evaluation)

from models import model_0,model_1, model_2, model_3, model_4, model_5
from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5

def get_full_dataset_with_targets(data_set_id, transform):
    if data_set_id == 1:  # CIFAR 10
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.cat((torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)))
    elif data_set_id == 2:  # CIFAR 100
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.cat((torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)))
    elif data_set_id == 3:  # MNIST
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        targets = torch.cat((torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)))
    else:
        raise ValueError(f"Invalid data_set_id: {data_set_id}")
    return full_dataset, targets


def run_cross_validation(config_filename, Kfolds, base_results_dir='results'):
    base_path = os.getcwd()
    config_path = os.path.join(base_path, "configs", config_filename)
    config = load_config(config_path)

    program_config = config["program"]
    data_config = config["data"]
    data_set_id = data_config["data_set_id"]
    model_config = config["model"]
    training_config = config["training"]

    seed = program_config["seed"]
    gen_rand_seed = program_config["gen_rand_seed"]
    gpu = program_config["gpu"]

    if gen_rand_seed:
        torch.manual_seed(datetime.now().microsecond % (2**32 - 1))
    else:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    model_id = model_config["model_id"]
    number_channels = model_config["number_channels"]
    number_classes = model_config["number_classes"]
    out_channels = model_config.get("out_channels")
    image_height = model_config.get("image_height")
    image_width = model_config.get("image_width")
    fc_hidden_size = model_config.get("fc_hidden_size", 128)
    fc_dropout_rate = model_config.get("fc_dropout_rate", 0.25)
    out_channels = model_config.get("out_channels", 32)
    num_layers = model_config.get("num_layers")

    batch_size = training_config["batch_size"]
    epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]

    loss_func = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset, targets = get_full_dataset_with_targets(data_set_id, transform)
    sample_image = full_dataset[0][0]
    image_height = sample_image.shape[1]
    image_width = sample_image.shape[2]

    skf = StratifiedKFold(n_splits=Kfolds, shuffle=True, random_state=seed)

    all_fold_metrics_raw = []

    print_section(f"{Kfolds}-Fold Cross-Validation")

    # Determine model name from config_filename for folder naming
    model_name = os.path.splitext(config_filename)[0] # e.g., 'model_1' from 'model_1.yaml'

    for fold, (train_indices, val_indices) in enumerate(skf.split(full_dataset, targets)):
        print_section(f"Fold {fold+1}/{Kfolds}")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

        # Model instantiation
        # ... (Your existing model instantiation logic remains here) ...
        if model_id == 0:
            model = model_0(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 1:
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
        # End of existing model instantiation logic

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

        # --- ADDED: Model saving path creation for each fold ---
        # Create a directory structure like: results/model_name/fold_X/
        model_fold_dir = os.path.join(base_results_dir, model_name, f"fold_{fold+1}")
        os.makedirs(model_fold_dir, exist_ok=True)
        model_save_path = os.path.join(model_fold_dir, f"{model_name}_fold_{fold+1}_params.pth")
        # --------------------------------------------------------

        last_epoch_metrics = train(
            model,
            train_loader,
            optimizer,
            loss_func,
            epochs,
            device,
            val_loader=val_loader,
            config_filename=config_filename,
            save_outputs=True, # Ensure this is True to enable saving
            model_save_path=model_save_path # Pass the specific path for this fold
        )

        last_epoch_metrics['Fold'] = fold + 1
        all_fold_metrics_raw.append(last_epoch_metrics)

    print_section("Cross-Validation Summary")

    df_metrics = pd.DataFrame(all_fold_metrics_raw)

    avg_train_loss = df_metrics['Train_Loss'].mean()
    avg_val_loss = df_metrics['Val_Loss'].mean()
    avg_train_acc = df_metrics['Train_Accuracy'].mean()
    avg_val_acc = df_metrics['Val_Accuracy'].mean()

    std_train_loss = df_metrics['Train_Loss'].std()
    std_val_loss = df_metrics['Val_Loss'].std()
    std_train_acc = df_metrics['Train_Accuracy'].std()
    std_val_acc = df_metrics['Val_Accuracy'].std()

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

    print(fold_df.to_string(index=False))

    sa_timezone = pytz.timezone('Africa/Johannesburg')
    current_time_sast = datetime.now(sa_timezone)
    current_time_str = current_time_sast.strftime("%Y%m%d_%H%M%S")
    yaml_base = os.path.splitext(config_filename)[0]

    # Changed export filename to reside within model's results directory
    cv_summary_dir = os.path.join(base_results_dir, model_name) # Summary will go inside the model's directory
    os.makedirs(cv_summary_dir, exist_ok=True)
    cv_export_filename = os.path.join(cv_summary_dir, f"crossv_summary_{yaml_base}_results_{current_time_str}.csv")

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
        "--Kfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5).",
    )
    args = parser.parse_args()

    run_cross_validation(args.config_file, Kfolds=args.Kfolds)