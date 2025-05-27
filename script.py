#! /usr/bin/python3

import argparse
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim

from utils import *
from models import *
from baseline_models import *
from train import *
from test import *

if __name__ == "__main__":
    # Accept config file as command parameter
    parser = argparse.ArgumentParser(description="Load YAML configuration file.")

    parser.add_argument('--config_file', type=str, help='Path to the configuration file')

    args = parser.parse_args()

    base_path = os.getcwd()

    config_path = os.path.join(base_path, "configs", args.config_file)

    config = load_config(config_path)

    # Allocate config variables
    program = config["program"]
    data = config["data"]
    model = config["model"]
    training = config["training"]

    # Program variables
    seed = program["seed"]
    gen_rand_seed = program["gen_rand_seed"]
    gpu = program["gpu"]

    # Data variables
    data_set_id = data["data_set_id"]
    test_ratio = data["test_ratio"]
    val_ratio = data["val_ratio"]

    # Model variables
    model_id = model["model_id"]
    number_channels = model["number_channels"]
    number_classes = model["number_classes"]

    # Training variables
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    learning_rate = training["learning_rate"]
    epsilon = training["epsilon"]
    grad_epsilon = training["grad_epsilon"]
    clip_factor = training["clip_factor"]
    # optimizer = training["optimizer"]
    # loss_func = training["loss_func"]

    print_section("Configuration")

    print(f"Seed: {seed}")
    print(f"Gen Rand Seed: {gen_rand_seed}")
    print(f"GPU: {gpu}")
    print(f"Data Set ID: {data_set_id}")
    print(f"Test Ratio: {test_ratio}")
    print(f"Validation Ratio: {val_ratio}")
    print(f"Model ID: {model_id}")
    print(f"Number of Channels: {number_channels}")
    print(f"Number of Classes: {number_classes}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epsilon: {epsilon}")
    print(f"Gradient Epsilon: {grad_epsilon}")
    print(f"Clip Factor: {clip_factor}")
    # print(f"optimizer: {optimizer}")
    # print(f"loss_func: {loss_func}")
   

    device = "cpu"
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Get number of workers for CPU
    num_workers = os.cpu_count()

    print_section("Data Downloading")

    if data_set_id == 1:
        # CIFAR10
        train_dataset_mean_std = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )

        mean, std = data_mean_std_rgb(train_dataset_mean_std)

        # Prepare data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform)

        test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform)

    elif data_set_id == 2:
        # CIFAR100
        train_dataset_mean_std = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )

        mean, std = data_mean_std_rgb(train_dataset_mean_std)

        # Prepare data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        train_dataset = datasets.CIFAR100(root="data", train=True, transform=transform)

        test_dataset = datasets.CIFAR100(root="data", train=False, transform=transform)

    elif data_set_id == 3:
        # MNIST
        train_dataset_mean_std = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )

        mean, std = data_mean_std_greyscale(train_dataset_mean_std)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        train_dataset = datasets.MNIST(root="data", train=True, transform=transform)

        test_dataset = datasets.MNIST(root="data", train=False, transform=transform)

    else:
        raise ValueError("Dataset ID not recognised")

    print_section("Train Dataset Mean and STD")

    print(f"Dataset Mean: {mean}")
    print(f"Dataset STD: {std}")

    image_shape = train_dataset[0][0].shape

    print(train_dataset[0][0].shape)

    dataset_summery(train_dataset, test_dataset)

    train_loader, val_loader = get_train_val_loaders(
        train_dataset, val_ratio, batch_size, num_workers, seed
    )

    test_loader = get_test_loader(test_dataset, batch_size, num_workers)

    print_section("Model")

    # model

    in_channels =  train_dataset[0][0].shape[0]
    out_channels = 32
    kernel_size = 3
    image_height = train_dataset[0][0].shape[1]
    image_width = train_dataset[0][0].shape[2]
    
    targets = [label for _, label in train_dataset]
    num_classes = len(set(targets))

    fc_hidden_size = 1
    fc_dropout_rate = 0.3
    num_layers = 1
    epsilon = 1e-8
    eps = 1e-3
    clip_factor = 0.1
    kernel_size = 3
    stride = 1
    padding = 1
    dropout_rate = 0.3
    learning_rate = 0.004
    loss_func = nn.CrossEntropyLoss()

    
    
    set_component_vars(epsilon, kernel_size, stride, padding, dropout_rate)

    if model_id == 1:
        model = model_1(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 2:
        model = model_2(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 3:
        model = model_3(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 4:
        model = model_4(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 5:
        model = model_5(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 6:
        model = baseline_model_1(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 7:
        model = baseline_model_2(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 8:
        model = baseline_model_3(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 9:
        model = baseline_model_4(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    elif model_id == 10:
        model = baseline_model_5(
            in_channels,
            out_channels,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
        )
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    else:
        print("Please input a valid model ID")

print(model)


print_section("Training")

train(model, train_loader, optimizer, loss_func, epochs, device,val_loader=val_loader)


print_section("Testing")

test(model, test_loader, loss_func, device)

print_section("Cross Validition")

# cross_validate(model, dataset, num_folds, batch_size, epochs, learning_rate, device, seed)


