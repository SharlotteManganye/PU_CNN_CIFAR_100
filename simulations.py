import os
import torch
import pytz
from datetime import datetime
from train import train
from test import test
from utils import *
from torchvision import datasets, transforms
from torch import nn, optim
from models import *
from baseline_models import *

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
    test_ratio = data["test_ratio"]
    val_ratio = data["val_ratio"]
    number_channels = model_cfg["number_channels"]
    number_classes = model_cfg["number_classes"]
    num_layers = model_cfg["num_layers"]
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    learning_rate = float(training["learning_rate"])
    epsilon = training["epsilon"]
    grad_epsilon = training["grad_epsilon"]
    clip_factor = training["clip_factor"]

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count()

    if data_set_id == 1:
        train_dataset_mean_std = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
        mean, std = data_mean_std_rgb(train_dataset_mean_std)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform)
    elif data_set_id == 2:
        train_dataset_mean_std = datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())
        mean, std = data_mean_std_rgb(train_dataset_mean_std)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR100(root="data", train=True, transform=transform)
        test_dataset = datasets.CIFAR100(root="data", train=False, transform=transform)
    elif data_set_id == 3:
        train_dataset_mean_std = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        mean, std = data_mean_std_greyscale(train_dataset_mean_std)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.MNIST(root="data", train=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, transform=transform)
    else:
        raise ValueError("Dataset ID not recognised")

    dataset_summery(train_dataset, test_dataset)

    train_loader, val_loader = get_train_val_loaders(train_dataset, val_ratio, batch_size, num_workers, seed)
    test_loader = get_test_loader(test_dataset, batch_size, num_workers)

    in_channels = train_dataset[0][0].shape[0]
    image_height = train_dataset[0][0].shape[1]
    image_width = train_dataset[0][0].shape[2]
    targets = [label for _, label in train_dataset]
    num_classes = len(set(targets))

    fc_hidden_size = 128
    fc_dropout_rate = 0.25
    dropout_rate = 0.25
    loss_func = nn.CrossEntropyLoss()

    for run_id in range(1, num_runs + 1):
        print_section(f"Simulation Run {run_id}")
        torch.manual_seed(seed + run_id)

        if model_id == 0:
            model = model_0(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 1:
            model = model_1(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 2:
            model = model_2(in_channels, 32, image_height, image_width, dropout_rate, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 3:
            model = model_3(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 4:
            model = model_4(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate, num_layers)
        elif model_id == 5:
            model = model_5(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate, num_layers)
        elif model_id == 6:
            model = baseline_model_1(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 7:
            model = baseline_model_2(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 8:
            model = baseline_model_3(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate)
        elif model_id == 9:
            model = baseline_model_4(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate, num_layers)
        elif model_id == 10:
            model = baseline_model_5(in_channels, 32, image_height, image_width, fc_hidden_size, number_classes, fc_dropout_rate, num_layers)
        else:
            raise ValueError("Invalid model ID")

       
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        
        yaml_base = os.path.splitext(os.path.basename(config_filename))[0]
        sim_dir = os.path.join(base_results_dir, yaml_base, f"model_{model_id}", f"simulation_{run_id}")
        model_save_path = os.path.join(sim_dir, "model_parameters", "final_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        epoch_metrics = train(
          model,
        train_loader,
        optimizer,
        loss_func,
        epochs,
        device,
        config_filename,
        val_loader=val_loader,
        base_results_dir=sim_dir,
        params_subdir="model_parameters",
        save_outputs=False  # Prevent `train()` from saving the CSV
        )
        sa_timezone = pytz.timezone('Africa/Johannesburg')
        current_time_str = datetime.now(sa_timezone).strftime("%Y%m%d_%H%M%S")


        df = pd.DataFrame(epoch_metrics)
        metrics_path = os.path.join(sim_dir, "model_parameters", f"epoch_metrics_{current_time_str}.csv")
        df.to_csv(metrics_path, index=False)
        print(f"Epoch metrics saved to {metrics_path}")

        test_loss, test_acc = test(model, test_loader, loss_func, device, config_filename, base_results_dir=os.path.join(sim_dir, "test"))
        print(f"Simulation {run_id} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

