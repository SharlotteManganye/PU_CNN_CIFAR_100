import os
import json
import torch
import csv
import pytz
from torch.utils.data import Subset
import random
from datetime import datetime
from functools import partial
import pyhopper as hp
from utils import get_train_val_loaders
from models import model_0, model_1, model_2, model_3, model_4, model_5
from baseline_models import (
    baseline_model_1, baseline_model_2, baseline_model_3,
    baseline_model_4, baseline_model_5
)

def get_subset(dataset, fraction=0.2, seed=42):
    size = len(dataset)
    subset_size = int(size * fraction)
    random.seed(seed)
    indices = random.sample(range(size), subset_size)
    return Subset(dataset, indices)

def train_for_hyperparam_search(model, train_loader, optimizer, loss_func, epochs, device, val_loader=None):
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

    with open(log_path, mode='a', newline='') as csvfile:
        fieldnames = list(params.keys()) + ['val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {**params, 'val_loss': val_loss}
        writer.writerow(row)

def objective(params, fixed_args):
    (
          model_id, in_channels, out_channels, kernel_size, stride, padding,
          dropout_rate, image_height, image_width, fc_hidden_size, number_classes,
          fc_dropout_rate, num_layers, train_dataset, val_ratio, seed, device,
          loss_func, epochs, base_results_dir
      ) = fixed_args


    current_batch_size = params["batch_size"]
    current_learning_rate = params["lr"]

    print(f"\n--- Hyperparameter Trial ---")
    print(f"Batch Size: {current_batch_size}, Learning Rate: {current_learning_rate}")

    model_constructors = {
        0: model_0, 1: model_1, 2: model_2, 3: model_3, 4: model_4, 5: model_5,
        6: baseline_model_1, 7: baseline_model_2, 8: baseline_model_3,
        9: baseline_model_4, 10: baseline_model_5
    }

    if model_id not in model_constructors:
        raise ValueError("Invalid model ID")

    model_args = [
        in_channels, out_channels, kernel_size, stride, padding, dropout_rate,
        image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate
    ]
    if model_id in [4, 5, 9, 10]:
        model_args.append(num_layers)

    model = model_constructors[model_id](*model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, weight_decay=1e-5)

    train_dataset_subset = get_subset(train_dataset, fraction=0.2, seed=seed)
    train_loader, val_loader = get_train_val_loaders(
        train_dataset_subset,
        batch_size=current_batch_size,
        val_ratio=val_ratio,
        num_workers=4,
        seed=seed
    )

    val_loss = train_for_hyperparam_search(
        model,
        train_loader,
        optimizer,
        loss_func,
        epochs,
        device,
        val_loader=val_loader
    )

    # Save intermediate result to CSV
    log_path = os.path.join(base_results_dir, "pyhopper", f"model_{model_id}", "intermediate_results.csv")
    log_trial_result_csv(log_path, params, val_loss)

    model = None
    torch.cuda.empty_cache()

    return val_loss

    

def run_hyperparameter_search(
          model_id, in_channels, out_channels, kernel_size, stride, padding,
          dropout_rate, image_height, image_width, fc_hidden_size, number_classes,
          fc_dropout_rate, num_layers, train_dataset, val_ratio, seed, device,
          loss_func, epochs, base_results_dir
    ):
    fixed_args = (
          model_id, in_channels, out_channels, kernel_size, stride, padding,
          dropout_rate, image_height, image_width, fc_hidden_size, number_classes,
          fc_dropout_rate, num_layers, train_dataset, val_ratio, seed, device,
          loss_func, epochs, base_results_dir
    )

    search_space = hp.Search({
        "batch_size": hp.int(32, 128, power_of=2),
        "lr": hp.float(1e-5, 1e-3, "0.1g"),
    })

    print("Starting Hyperparameter Search...")
    results = search_space.run(
        partial(objective, fixed_args=fixed_args),
        "minimize",
        "4h",
        n_jobs=4
    )
    print("Hyperparameter Search Completed.")

    if hasattr(results, 'best_f') and results.best_f is not None:
        best_params = results.best_params
        best_loss = results.best_f

        sa_timezone = pytz.timezone('Africa/Johannesburg')
        current_time_sast = datetime.now(sa_timezone)
        current_timestamp = current_time_sast.strftime("%Y%m%d_%H%M%S")

        model_dir_name = f"model_{model_id}"
        optimization_results_dir = os.path.join(base_results_dir, "pyhopper", model_dir_name)
        os.makedirs(optimization_results_dir, exist_ok=True)

        optimization_filename = os.path.join(
            optimization_results_dir,
            f"model{model_id}_optimization.json"
        )

        data_to_save = {
            "model_id": model_id,
            "config_file_used_for_search": config_filename,
            "optimal_parameters": best_params,
            "best_validation_loss": best_loss
        }

        try:
            with open(optimization_filename, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Optimal parameters for Model {model_id} saved to: {optimization_filename}")
        except Exception as e:
            print(f"Error saving optimal parameters for Model {model_id}: {e}")
    else:
        print("No optimal parameters found or search did not return expected results.")
        print(f"Debug: Results object: {results}")

