import yaml
import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter
import os
import pandas as pd
from datetime import datetime




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

def save_model_checkpoint(model, epoch, params_dir):

    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    checkpoint_filename = os.path.join(params_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_filename)
    return checkpoint_filename


def training_results(all_epoch_metrics, model, current_time_str, base_results_dir='results', params_subdir='model_parameters'):


    # Create a unique timestamped folder for this training run's results
    run_results_dir = os.path.join(base_results_dir, f'run_{current_time_str}')
    os.makedirs(run_results_dir, exist_ok=True)
    
    # Create directory for saving parameters within the run-specific results directory
    params_dir = os.path.join(run_results_dir, params_subdir)
    os.makedirs(params_dir, exist_ok=True)

    # Now, iterate through the collected metrics and add checkpoint paths
    final_epoch_data = []
    for epoch_metric in all_epoch_metrics:
        epoch = epoch_metric['Epoch']
        # Save model checkpoint and get its path
        checkpoint_path = save_model_checkpoint(model, epoch, params_dir)
        epoch_metric['Checkpoint_Path'] = checkpoint_path # Add checkpoint path to the metrics
        final_epoch_data.append(epoch_metric)

    # Save all collected data (including checkpoint paths) to a CSV file
    metrics_csv_filename = os.path.join(run_results_dir, f'training_metrics_{current_time_str}.csv')
    df = pd.DataFrame(final_epoch_data)
    df.to_csv(metrics_csv_filename, index=False)
    
    print(f"Training metrics (including batch size and LR) saved to {metrics_csv_filename}")
    print(f"Model checkpoints saved to {params_dir}")



def test_results(test_loss, test_accuracy, base_results_dir='results'):
    sa_timezone = pytz.timezone('Africa/Johannesburg') 
    current_time_sast = datetime.now(sa_timezone)
    current_time_str = current_time_sast.strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_results_dir, exist_ok=True)
    test_metrics_filename = os.path.join(base_results_dir, f'test_metrics_{current_time_str}.csv')
    test_data = {
        'Timestamp': current_time_str,
        'Test_Loss': test_loss,
        'Test_Accuracy': test_accuracy
    }
    df = pd.DataFrame([test_data]) 
    df.to_csv(test_metrics_filename, index=False) 
    print(f"Test results saved to {test_metrics_filename}")
