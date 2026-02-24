# /content/drive/MyDrive/PU_CNN/Convolutional-Neural-Networks/visualize_feature_maps.py

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import yaml
import argparse # Needed if you keep the __main__ block for standalone testing

# --- Import your model definitions ---
# Ensure these paths are correct relative to where this script is run or in your PYTHONPATH
from models import model_0, model_1, model_2, model_3, model_4, model_5,PI_ResNet,PI_ResNet_stacked,PU_ResNet18

from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5,baseline_model_6,baseline_model_7

# Assuming utils.py contains load_config and print_section
from utils import load_config, print_section

# --- The save_feature_maps_from_model function (as provided before) ---
def save_feature_maps_from_model(model, loader, mean, std, device, model_name, output_base_dir="results/feature_maps", num_maps=16):
    """
    Saves feature maps from a given model for a sample image.
    (Full implementation as previously provided)
    """
    # Set up output directory
    output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load one batch of data
    model.eval()
    model.to(device)
    data_iter = iter(loader)
    image, label = next(data_iter)
    image = image.to(device)

    with torch.no_grad():
        output_and_features = model(image, return_feature_maps=True)

    output = output_and_features[0]
    feature_maps_list = output_and_features[1:]

    img_np = image[0].cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)

    img_np = np.transpose(img_np, (1, 2, 0)) * np.array(std) + np.array(mean)
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(3, 3))
    plt.imshow(img_np)
    plt.axis('off')
    plt.title("Original Image")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "original_image.png"))
    plt.close()

    for layer_idx, feature_maps in enumerate(feature_maps_list):
        if not isinstance(feature_maps, torch.Tensor) or feature_maps.numel() == 0:
            print(f"Skipping empty or invalid feature map for layer {layer_idx + 1}")
            continue

        layer_dir = os.path.join(output_dir, f"Conv{layer_idx+1}_Features")
        os.makedirs(layer_dir, exist_ok=True)
        num_maps_to_save = min(num_maps, feature_maps.shape[1])

        print(f"Saving {num_maps_to_save} feature maps for Layer {layer_idx + 1}...")
        for i in range(num_maps_to_save):
            fmap = feature_maps[0, i].cpu().detach().numpy()
            plt.imshow(fmap, cmap='viridis')
            plt.axis('off')
            plt.title(f"Feature Map {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f"Feature_{i}.png"))
            plt.close()

    print(f"Feature maps saved to {output_dir}")

# --- The get_full_dataset_with_targets function (as provided before) ---
def get_full_dataset_with_targets(data_set_id):
    """
    Loads the full dataset with specific augmentations and normalization.
    """
    # Placeholder functions for mean/std calculation 
    # (Ensure these are defined in your utils or script)
    def get_stats(dataset, is_rgb=True):
        if is_rgb:
            return data_mean_std_rgb(dataset)
        return data_mean_std_greyscale(dataset)

    if data_set_id == 1:  # CIFAR 10
        raw_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mean, std = get_stats(raw_data, is_rgb=True)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        targets = torch.cat((torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)))

    elif data_set_id == 2:  # CIFAR 100
        raw_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mean, std = get_stats(raw_data, is_rgb=True)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        targets = torch.cat((torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)))

    elif data_set_id == 3:  # MNIST
        raw_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mean, std = get_stats(raw_data, is_rgb=False)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        targets = torch.cat((train_dataset.targets, test_dataset.targets))

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    return full_dataset, targets, mean, std

# --- Main function to run the feature map saving process ---
def run_feature_map_saver(config_filename, fold_number=1, num_maps=16):
    """
    Loads a trained model based on config and saves its feature maps.
    This function encapsulates all the logic and takes config_filename as an argument.
    """
    base_path = os.getcwd()
    config_path = os.path.join(base_path, "configs", config_filename)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    config = load_config(config_path)

    # ... (Rest of the model/data setup logic) ...
    data_config = config["data"]
    data_set_id = data_config["data_set_id"]
    model_config = config["model"]
    training_config = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = model_config["model_id"]
    number_channels = model_config["number_channels"]
    number_classes = model_config["number_classes"]
    out_channels = model_config.get("out_channels")
    fc_hidden_size = model_config.get("fc_hidden_size", 128)
    fc_dropout_rate = model_config.get("fc_dropout_rate", 0.25)
    num_layers = model_config.get("num_layers")

    batch_size = training_config["batch_size"]

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset, _, dataset_mean, dataset_std = get_full_dataset_with_targets(data_set_id, transform)
    sample_image = full_dataset[0][0]
    image_height = sample_image.shape[1]
    image_width = sample_image.shape[2]

    # Instantiate the correct model based on model_id
    print(f"Instantiating Model ID: {model_id}...")

    if model_id == 1:
        model = model_1(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 2:
        model = model_2(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 3:
        model = model_3(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(learning_rate), weight_decay=weight_decay
        )

    elif model_id == 0:
        model = model_0(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 4:
        model = model_4(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 5:
        model = model_5(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
            image_height,
            image_width,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 6:
        model = baseline_model_1(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 8:
        model = baseline_model_3(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_rate,
            number_classes,
            fc_dropout_rate,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif model_id == 11:   
        model = PI_ResNet()
        optimizer =optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    elif model_id == 12:
        model = PI_ResNet_stacked()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif model_id == 13:
     
        model = PU_ResNet18(ResidualBlock,ResidualBlock_PU)
        optimizer =optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif model_id == 14:
     
        model = baseline_model_6()
        optimizer =optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    elif model_id == 15:
     
        model = baseline_model_7()
        optimizer =optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError(f"Invalid model ID: {model_id}")

    model = model.to(device)

    # Determine model name from config_filename (NOW INSIDE THE FUNCTION)
    model_name = os.path.splitext(config_filename)[0]

    # Construct the path to the saved model parameters
    model_load_path = os.path.join("results", model_name, f"fold_{fold_number}", f"{model_name}_fold_{fold_number}_params.pth")

    if not os.path.exists(model_load_path):
        print(f"Error: Model parameters not found for fold {fold_number} at {model_load_path}")
        print("Please ensure the cross-validation script has been run and saved the model for this fold.")
        return

    print(f"Loading model parameters from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device))

    # Create a DataLoader for a single sample (or a small batch)
    sample_loader = DataLoader(Subset(full_dataset, [0]), batch_size=1, shuffle=False)

    print(f"Saving feature maps for model: {model_name} (from fold {fold_number})")
    save_feature_maps_from_model(
        model=model,
        loader=sample_loader,
        mean=dataset_mean,
        std=dataset_std,
        device=device,
        model_name=model_name,
        num_maps=num_maps
    )

# This block only runs when visualize_feature_maps.py is executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save feature maps for a trained model.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML config file (e.g., 'model_1.yaml')."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="The fold number of the model to load (default: 1).",
    )
    parser.add_argument(
        "--num_maps",
        type=int,
        default=16,
        help="Number of feature maps to save per layer (default: 16).",
    )
    args = parser.parse_args()

    run_feature_map_saver(args.config_file, fold_number=args.fold, num_maps=args.num_maps)