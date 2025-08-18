import os
import json
import torch
import csv
import pytz
from torch.utils.data import Subset
import random
import itertools
from datetime import datetime
from functools import partial
import pyhopper as hp
from utils import get_train_val_loaders

from model_components import ResidualBlock

from models import model_0, model_1, model_2, model_3, model_4, model_5, ResNet
from baseline_models import (
    baseline_model_1,
    baseline_model_2,
    baseline_model_3,
    baseline_model_4,
    baseline_model_5,
)


def get_subset(dataset, fraction=0.2, seed=42):
    size = len(dataset)
    subset_size = int(size * fraction)
    random.seed(seed)
    indices = random.sample(range(size), subset_size)
    return Subset(dataset, indices)


def train_for_hyperparam_search(
    model, train_loader, optimizer, loss_func, epochs, device, val_loader=None
):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def log_trial_result_csv(log_path, params, val_loss):
    file_exists = os.path.isfile(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, mode="a", newline="") as csvfile:
        fieldnames = list(params.keys()) + ["val_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {**params, "val_loss": val_loss}
        writer.writerow(row)


def objective(params, fixed_args):
    (
        model_id,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dropout_rate,
        image_height,
        image_width,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
        num_layers,
        train_dataset,
        val_ratio,
        seed,
        device,
        loss_func,
        epochs,
        base_results_dir,
    ) = fixed_args

    current_batch_size = params["batch_size"]
    current_learning_rate = params["lr"]

    print(f"\n--- Hyperparameter Trial ---")
    print(f"Batch Size: {current_batch_size}, Learning Rate: {current_learning_rate}")

    model_constructors = {
        0: model_0,
        1: model_1,
        2: model_2,
        3: model_3,
        4: model_4,
        5: model_5,
        11: ResNet,
        6: baseline_model_1,
        7: baseline_model_2,
        8: baseline_model_3,
        9: baseline_model_4,
        10: baseline_model_5,
    }

    if model_id not in model_constructors:
        raise ValueError("Invalid model ID")

    model_args = [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dropout_rate,
        number_classes,
        fc_dropout_rate,
    ]
    if model_id in [0, 3]:

        model_args = [
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        ]
    elif model_id == 11:
        model_args = [ResidualBlock, number_classes]
    elif model_id in [4, 5]:
        model_args = [
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        ]
    elif model_id in [9, 10]:
        # if baseline models 9 and 10 do not need image dims, keep as is:
        model_args.append(num_layers)

    model = model_constructors[model_id](*model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=current_learning_rate, weight_decay=1e-5
    )
    num_workers = min(4, os.cpu_count())
    train_dataset_subset = get_subset(train_dataset, fraction=0.2, seed=seed)
    train_loader, val_loader = get_train_val_loaders(
        train_dataset_subset,
        batch_size=current_batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        seed=seed,
    )

    val_loss = train_for_hyperparam_search(
        model, train_loader, optimizer, loss_func, epochs, device, val_loader=val_loader
    )

    # Save intermediate result to CSV
    log_path = os.path.join(
        base_results_dir, "pyhopper", f"model_{model_id}", "intermediate_results.csv"
    )
    log_trial_result_csv(log_path, params, val_loss)

    model = None
    torch.cuda.empty_cache()

    return val_loss


def run_hyperparameter_search_grid(
    model_id,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dropout_rate,
    image_height,
    image_width,
    fc_hidden_size,
    number_classes,
    fc_dropout_rate,
    num_layers,
    train_dataset,
    val_ratio,
    seed,
    device,
    loss_func,
    epochs,
    base_results_dir,
    config_filename=None,
):
    # Define grid search space
    batch_sizes = [32, 64, 128]
    learning_rates = [1e-5, 1e-4, 1e-3]

    param_grid = list(itertools.product(batch_sizes, learning_rates))

    best_val_loss = float("inf")
    best_params = None

    sa_timezone = pytz.timezone("Africa/Johannesburg")
    current_time_sast = datetime.now(sa_timezone)
    current_timestamp = current_time_sast.strftime("%Y%m%d_%H%M%S")

    model_dir_name = f"model_{model_id}"
    optimization_results_dir = os.path.join(
        base_results_dir, "gridsearch", model_dir_name
    )
    os.makedirs(optimization_results_dir, exist_ok=True)
    log_path = os.path.join(optimization_results_dir, "intermediate_results.csv")

    for batch_size, lr in param_grid:
        params = {"batch_size": batch_size, "lr": lr}
        print(f"\n--- Grid Search Trial ---")
        print(f"Batch Size: {batch_size}, Learning Rate: {lr}")

        fixed_args = (
            model_id,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
            train_dataset,
            val_ratio,
            seed,
            device,
            loss_func,
            epochs,
            base_results_dir,
        )

        try:
            val_loss = objective(params, fixed_args)
        except Exception as e:
            print(f"Trial failed for params {params}: {e}")
            continue

        log_trial_result_csv(log_path, params, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    if best_params:
        optimization_filename = os.path.join(
            optimization_results_dir, f"model{model_id}_gridsearch_optimization.json"
        )

        data_to_save = {
            "model_id": model_id,
            "config_file_used_for_search": config_filename,
            "optimal_parameters": best_params,
            "best_validation_loss": best_val_loss,
        }

        try:
            with open(optimization_filename, "w") as f:
                json.dump(data_to_save, f, indent=4)
            print(
                f"Optimal parameters for Model {model_id} saved to: {optimization_filename}"
            )
        except Exception as e:
            print(f"Error saving optimal parameters for Model {model_id}: {e}")
    else:
        print("Grid search completed, but no valid trial succeeded.")
