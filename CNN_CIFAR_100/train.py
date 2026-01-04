import torch
from datetime import datetime
import pytz
import os
import pandas as pd

def adaptive_clip_grad_norm(parameters, clip_factor=0.01, eps=1e-3):
    if not isinstance(parameters, torch.Tensor):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters:
        return 0.0
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
    clip_coef = (clip_factor * total_norm) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm.item()

def train(model, train_loader, optimizer, loss_func, device, epoch):
    """
    Trains the model for a single epoch and returns average loss and accuracy.
    """
    model.train()
    train_loss = 0
    train_acc = 0
    
    # You can remove the print statements here as the simulations file will handle it.
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.float())
        loss = loss_func(output, target)
        loss.backward()
        grad_norm = adaptive_clip_grad_norm(model.parameters())
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, pred = output.max(1)
        train_acc += target.eq(pred).sum().item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = 100. * train_acc / len(train_loader.dataset)

    # Return loss and accuracy for the single epoch
    return avg_train_loss, avg_train_acc

# You can place your `test` function here and rename it to `test_epoch`
# to match the naming convention.
def test(model, test_loader, loss_func, device):
    """
    Evaluates the model on the test set and returns loss and accuracy.
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = loss_func(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = output.max(1)
            test_acc += target.eq(pred).sum().item()
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_acc = 100. * test_acc / len(test_loader.dataset)
    return avg_test_loss, avg_test_acc