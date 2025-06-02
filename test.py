def test(model, test_loader, loss_func, device, yaml_filename, base_results_dir='results'):
    """
    Tests the model and saves the test loss and accuracy to a CSV file.
    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_func (callable): Loss function.
        device (torch.device): Device to test on (e.g., 'cuda' or 'cpu').
        yaml_filename (str): Name of the YAML file used for the model configuration.
        base_results_dir (str): Base directory where test results will be saved.
    """
    import torch
    import os
    import pandas as pd
    from datetime import datetime
    import pytz

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

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%')

    # Time and filename formatting
    sa_timezone = pytz.timezone('Africa/Johannesburg')
    current_time_str = datetime.now(sa_timezone).strftime("%Y%m%d_%H%M%S")
    yaml_base = os.path.splitext(os.path.basename(yaml_filename))[0]
    filename = f'test_{yaml_base}_{current_time_str}.csv'

    os.makedirs(base_results_dir, exist_ok=True)
    test_metrics_filename = os.path.join(base_results_dir, filename)

    # Save results
    test_data = {
        'Timestamp': current_time_str,
        'YAML_File': yaml_filename,
        'Test_Loss': avg_test_loss,
        'Test_Accuracy': avg_test_acc
    }
    pd.DataFrame([test_data]).to_csv(test_metrics_filename, index=False)

    print(f"Test results saved to {test_metrics_filename}")
    return avg_test_loss, avg_test_acc
