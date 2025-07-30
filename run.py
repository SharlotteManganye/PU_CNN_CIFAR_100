#! /usr/bin/python3

import argparse
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
import re



from utils import *
from models import *
from baseline_models import *
from train import *
from test import *
from grid_search import *
from hyperparameter_search import run_hyperparameter_search
from cross_validation import *
from visualize_feature_maps import * 
from simulations import *

if __name__ == "__main__":
    # Accept config file as command parameter
    parser = argparse.ArgumentParser(description="Load YAML configuration file.")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="model_2.yaml",
        type=str,
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    config_filename = args.config_file

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
    num_layers = model["num_layers"]

    # Training variables
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    learning_rate = training["learning_rate"]
    epsilon = training["epsilon"]
    grad_epsilon = training["grad_epsilon"]
    clip_factor = training["clip_factor"]

    Kfolds = training["Kfolds"]
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
    print(f"Number of Layers: {num_layers}")
    print(f"Kfolds: {Kfolds}")
    # print(f"loss_func: {loss_func}")
   


    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
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

    # Data augmentation for training
      train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      train_dataset = datasets.CIFAR10(root="data", train=True, transform=train_transform)
      test_dataset = datasets.CIFAR10(root="data", train=False, transform=test_transform)

    elif data_set_id == 2:
    # CIFAR100
      train_dataset_mean_std = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
      )

      mean, std = data_mean_std_rgb(train_dataset_mean_std)

      train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      train_dataset = datasets.CIFAR100(root="data", train=True, transform=train_transform)
      test_dataset = datasets.CIFAR100(root="data", train=False, transform=test_transform)

    elif data_set_id == 3:
      # MNIST
      train_dataset_mean_std = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
      )

      mean, std = data_mean_std_greyscale(train_dataset_mean_std)

      train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

      train_dataset = datasets.MNIST(root="data", train=True, transform=train_transform)
      test_dataset = datasets.MNIST(root="data", train=False, transform=test_transform)

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
    out_channels = 16
    kernel_size = 3
    image_height = train_dataset[0][0].shape[1]
    image_width = train_dataset[0][0].shape[2]
    targets = [label for _, label in train_dataset]
    num_classes = len(set(targets))

    fc_hidden_size = 128
    fc_dropout_rate = 0.25
    epsilon = 1e-10
    eps = 1e-3
    clip_factor =  0.01
    kernel_size = 3
    stride = 1
    padding = 1
    dropout_rate = 0.25
    learning_rate = 1e-3
    loss_func = nn.CrossEntropyLoss()

    
    
    set_component_vars(epsilon, kernel_size, stride, padding, dropout_rate)

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
        # fc_input_size,
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
model_name_str = os.path.splitext(config_filename)[0]
print(model)


# print_section("Training")

# train(model, train_loader, optimizer, loss_func, epochs, device, config_filename, val_loader=val_loader, base_results_dir='results', params_subdir='model_parameters')

# print_section("Testing")

# test_loss, test_acc = test(model, test_loader, loss_func, device, config_filename, base_results_dir='results/test', save_results=True, epoch=epochs)


# print_section("SIMULATIONS")

# run_simulations(config_filename)


# print_section("Hyperparameter Search")


# run_hyperparameter_search(
#         model_id=model_id,
#         in_channels=number_channels,
#         out_channels=out_channels,
#         number_classes=number_classes,
#         image_height=image_height,
#         image_width=image_width,
#         fc_hidden_size=fc_hidden_size,
#         fc_dropout_rate=fc_dropout_rate,
#         num_layers=num_layers,
#         epochs=epochs,
#         val_ratio=val_ratio,
#         seed=seed,
#         device=device,
#         loss_func=loss_func,
#         train_dataset=train_dataset,
#         base_results_dir= 'results',
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         dropout_rate=dropout_rate
#     )



print_section("Grid_Search")

run_hyperparameter_search_grid(
    model_id=model_id, 
    in_channels=number_channels,
    out_channels=out_channels,
    kernel_size=kernel_size, 
    stride=stride, 
    padding=padding,
    dropout_rate=dropout_rate,
    image_height=image_height,
    image_width=image_width, 
    fc_hidden_size=fc_hidden_size, 
    number_classes=number_classes,
    fc_dropout_rate=fc_dropout_rate, 
    num_layers=num_layers, 
    train_dataset=train_dataset, 
    val_ratio=val_ratio, 
    seed=seed, 
    device=device,
    loss_func=loss_func, 
    epochs=epochs, 
    base_results_dir='results', 
    config_filename=config_filename
)


# print_section("Cross Validition")

# run_cross_validation(config_filename,Kfolds, base_results_dir='results' )

# print_section("Feature Maps Visualization")
# save_feature_maps_from_model(model, train_loader, mean, std, device, model_name=model_name_str, output_base_dir="results/feature_maps", num_maps=16)


