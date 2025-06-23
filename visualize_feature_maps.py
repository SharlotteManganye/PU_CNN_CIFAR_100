# visualize_feature_maps.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def show_feature_maps(feature_maps, title, num_maps=6):
    num_maps = min(num_maps, feature_maps.shape[1])
    plt.figure(figsize=(15, 3))
    for i in range(num_maps):
        plt.subplot(1, num_maps, i + 1)
        plt.imshow(feature_maps[0, i].cpu().detach().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'{title} [{i}]')
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, loader, mean, std, device, num_maps=6):
    model.eval()
    model.to(device)
    data_iter = iter(loader)
    image, label = next(data_iter)
    image = image.to(device)

    with torch.no_grad():
        output, conv1_features, conv2_features = model(image, return_feature_maps=True)

    # Show original image
    img_np = image[0].cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0)) * np.array(std) + np.array(mean)
    img_np = np.clip(img_np, 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Show feature maps
    show_feature_maps(conv1_features, "Conv1 Features", num_maps)
    show_feature_maps(conv2_features, "Conv2 Features", num_maps)
