import os
import torch
import pytz
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torch import nn, optim
from train import train
from test import test
from utils import *
from models import *
from baseline_models import *

def get_train_loader(train_dataset, batch_size, num_workers, seed):
    torch.manual_seed(seed)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def run_simulations(config_filename="model_2.yaml", model_id=1, num_runs=5, base_results_dir="results/simulations"):
    config_path = os.path.join("configs", config_filename)
    config = load_config(config_path)

    program = config["program"]
    data = config["data"]
    model_cfg = config["model"]
    training = config["training"]

    seed = program["seed"]
    gpu = program["gpu"]
    data_set_id = data["data_set_id"]
    val_ratio = data["val_ratio"]
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    learning_rate = float(training["learning_rate"])
    number_classes = model_cfg["number_classes"]
    num_layers = model_cfg.get("num_layers", None) 

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    num_workers = 2 

    if data_set_id == 1:
        dataset = datasets.CIFAR10
        mean_std_func = data_mean_std_rgb
    elif data_set_id == 2:
        dataset = datasets.CIFAR100
        mean_std_func = data_mean_std_rgb
    elif data_set_id == 3:
        dataset = datasets.MNIST
        mean_std_func = data_mean_std_greyscale
    else:
        raise ValueError("Dataset ID not recognised")

    train_dataset_mean_std = dataset(root="./data", train=True, download=True, transform=transforms.ToTensor())
    mean, std = mean_std_func(train_dataset_mean_std)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    full_train_dataset = dataset(root="data", train=True, transform=transform)
    test_dataset = dataset(root="data", train=False, transform=transform)

    val_size = int(val_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    new_train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    combined_test_dataset = ConcatDataset([val_dataset, test_dataset])

    train_loader = get_train_loader(new_train_dataset, batch_size, num_workers, seed)
    test_loader = get_test_loader(combined_test_dataset, batch_size, num_workers)

    in_channels = full_train_dataset[0][0].shape[0]
    image_height = full_train_dataset[0][0].shape[1]
    image_width = full_train_dataset[0][0].shape[2]

    fc_hidden_size = 128
    fc_dropout_rate = 0.25
    dropout_rate = 0.25 
    loss_func = nn.CrossEntropyLoss()

    for run_id in range(1, num_runs + 1):
        print_section(f"Simulation Run {run_id}")
        torch.manual_seed(seed + run_id)

        model = select_model(model_id, in_channels, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate, dropout_rate, num_layers)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model_name = os.path.splitext(config_filename)[0]
        sim_dir = os.path.join(base_results_dir, model_name, f"simulation_{run_id}")
        os.makedirs(sim_dir, exist_ok=True)

        epoch_metrics = [] 
        for epoch in range(1, epochs + 1):
            current_epoch_train_metrics_list = train(
                model, train_loader, optimizer, loss_func, 1, device, config_filename,
                val_loader=None, 
                base_results_dir=sim_dir,
                params_subdir="",
                save_outputs=False 
            )

            current_train_metrics = current_epoch_train_metrics_list[0] 
            
            train_loss_current_epoch = current_train_metrics['Train_Loss']
            train_acc_current_epoch = current_train_metrics['Train_Accuracy']

            test_loss, test_acc = test(model, test_loader, loss_func, device, config_filename, base_results_dir=sim_dir)
            
            # Use the extracted scalar values for printing
            print(f"Epoch {epoch}: Train Loss = {train_loss_current_epoch:.4f}, Train Accuracy = {train_acc_current_epoch:.2f}%, Test Accuracy = {test_acc:.2f}%, Test Loss = {test_loss:.4f}")
            epoch_metrics.append({
                "Epoch": epoch,
                "Train_Loss": train_loss_current_epoch,
                "Train_Accuracy": train_acc_current_epoch,
                "Test_Accuracy": test_acc,
                "Test_Loss": test_loss
            })

        sa_timezone = pytz.timezone('Africa/Johannesburg')
        current_time_str = datetime.now(sa_timezone).strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(epoch_metrics)
        metrics_path = os.path.join(sim_dir, f"epoch_metrics.csv") 
        df.to_csv(metrics_path, index=False)
        print(f"Epoch metrics saved to {metrics_path}")

def select_model(model_id, in_channels, height, width, fc_hidden_size, num_classes, fc_dropout, dropout, num_layers):

    if model_id == 1:
        model = model_1(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 2:
        model = model_2(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 3:
        model = model_3(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 0:
        model = model_0(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 4:
        model = model_4(
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

        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 5:
        model = model_5(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 6:
        model = baseline_model_1(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 7:
        model = baseline_model_2(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 8:
        model = baseline_model_3(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 9:
        model = baseline_model_4(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    elif model_id == 10:
        model = baseline_model_5(
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
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

    else:
        print("Please input a valid model ID")

