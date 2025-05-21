import yaml
import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def print_section(title=""):
    line = "-" * 50
    if title:
        print(f"\n{line}\n{title.center(50)}\n{line}\n")
    else:
        print(f"\n{line}\n")


def data_mean_std_greyscale(trainset):

    imgs = [item[0] for item in trainset]
    imgs = torch.stack(imgs, dim=0).numpy()

    mean = imgs.mean()
    std = imgs.std()

    return mean, std


def data_mean_std_rgb(trainset):

    imgs = [item[0] for item in trainset]
    imgs = torch.stack(imgs, dim=0).numpy()

    mean_r = imgs[:, 0, :, :].mean()
    mean_g = imgs[:, 1, :, :].mean()
    mean_b = imgs[:, 2, :, :].mean()

    std_r = imgs[:, 0, :, :].std()
    std_g = imgs[:, 1, :, :].std()
    std_b = imgs[:, 2, :, :].std()

    mean = [mean_r, mean_g, mean_b]
    std = [std_r, std_g, std_b]

    return mean, std


def dataset_summery(train_dataset, test_dataset):
    num_samples = len(train_dataset)
    image_size = train_dataset[0][0].size()
    class_distribution = Counter(label for _, label in train_dataset)
    sorted_class_distribution = dict(sorted(class_distribution.items()))

    print_section("Train Dataset")

    print(f"Number of samples: {num_samples}")
    print(f"Image size: {image_size}")
    print(f"Class distribution: {sorted_class_distribution}")

    num_samples = len(test_dataset)
    image_size = test_dataset[0][0].size()
    class_distribution = Counter(label for _, label in test_dataset)
    sorted_class_distribution = dict(sorted(class_distribution.items()))

    print_section("Test Dataset")

    print(f"Number of samples: {num_samples}")
    print(f"Image size: {image_size}")
    print(f"Class distribution: {sorted_class_distribution}")


def get_train_val_loaders(train_dataset, val_ratio, batch_size, num_workers, seed):
    total_train_size = len(train_dataset)
    val_size = int(val_ratio * total_train_size)
    train_size = total_train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_test_loader(test_dataset, batch_size, num_workers):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader


def create_run():
    # Create the run folder
    # Save yaml parameters
    print()


def training_results():
    # Append a row to a trining results .csv file
    print()


def test_results():
    # save final accuracy to separte test .csv file
    print()
