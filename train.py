
import torch
from datetime import datetime
import pytz
from utils import training_results # Ensure this is still relevant for metric logging, or remove if saving is handled externally
import os

def val(model, val_loader, loss_func, device):
    """
    Evaluates the model on the validation set and returns loss and accuracy.
    """
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = loss_func(output, target)
            val_loss += loss.item() * data.size(0)
            _, pred = output.max(1)
            val_acc += target.eq(pred).sum().item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = 100. * val_acc / len(val_loader.dataset)
    return avg_val_loss, avg_val_acc

def adaptive_clip_grad_norm(parameters, clip_factor=0.01, eps=1e-3):
    if not isinstance(parameters, torch.Tensor):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters:
        return 0.0  # Return 0 if no gradients
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
    clip_coef = (clip_factor * total_norm) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm.item()



def train(model, train_loader, optimizer, loss_func, epochs, device, config_filename, val_loader,
          base_results_dir='results', params_subdir='model_parameters', save_outputs=True,
          model_save_path=None): # ADDED: model_save_path argument
    model.train()
    batch_size = train_loader.batch_size if train_loader else "N/A"
    learning_rate = optimizer.param_groups[0]['lr'] if optimizer.param_groups else "N/A"
    sa_timezone = pytz.timezone('Africa/Johannesburg')
    current_time_sast = datetime.now(sa_timezone)
    current_time_str = current_time_sast.strftime("%Y%m%d_%H%M%S")
    all_epoch_metrics = []
    print(f"Training started for run at {current_time_str} SAST.")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
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
        avg_val_loss = 0.0
        avg_val_acc = 0.0

        if val_loader:
            avg_val_loss, avg_val_acc = val(model, val_loader, loss_func, device)

        epoch_data = {
            'Epoch': epoch + 1,
            'Train_Loss': avg_train_loss,
            'Train_Accuracy': avg_train_acc, 
            'Val_Loss': avg_val_loss,
            'Val_Accuracy': avg_val_acc,
            'Batch_Size': batch_size,
            'Learning_Rate': learning_rate,
            'Gradient_Norm': grad_norm,
        }
        all_epoch_metrics.append(epoch_data)
        print(f'Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.2f}%')
        if val_loader:
            print(f'\tVal Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%')

    if save_outputs and model_save_path: 

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model parameters saved to {model_save_path}")
    elif save_outputs and not model_save_path:
        print("Warning: save_outputs is True but model_save_path was not provided. Model parameters not saved.")


    return all_epoch_metrics