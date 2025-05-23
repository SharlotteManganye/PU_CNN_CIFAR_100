import torch
import os
import pandas as pd
from datetime import datetime
import pytz # Import pytz for timezone handling

def test(model, test_loader, loss_func, device, base_results_dir='results'):
    """
    Tests the model and saves the test loss and accuracy to a CSV file.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_func (callable): Loss function.
        device (torch.device): Device to test on (e.g., 'cuda' or 'cpu').
        base_results_dir (str): Base directory where test results will be saved.
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = loss_func(output, target)
            test_loss += loss.item() * data.size(0) # Multiply by batch size for correct average
            _, pred = output.max(1)
            test_acc += target.eq(pred).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset) # Divide by dataset size
    avg_test_acc = 100. * test_acc / len(test_loader.dataset)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%')

    # Define South Africa timezone
    sa_timezone = pytz.timezone('Africa/Johannesburg')

    current_time_sast = datetime.now(sa_timezone)
 
    current_time_str = current_time_sast.strftime("%Y%m%d_%H%M%S")

    # Ensure the base results directory exists
    os.makedirs(base_results_dir, exist_ok=True)

    # Define the path for the test metrics CSV file
    test_metrics_filename = os.path.join(base_results_dir, f'test_metrics_{current_time_str}.csv')

    # Prepare data for DataFrame
    test_data = {
        'Timestamp': current_time_str,
        'Test_Loss': avg_test_loss,
        'Test_Accuracy': avg_test_acc
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame([test_data]) # Wrap in list because it's a single row
    df.to_csv(test_metrics_filename, index=False)
    
    print(f"Test results saved to {test_metrics_filename}")

    return avg_test_loss, avg_test_acc

# Sample usage (place this in your main script or Colab cell)
# Make sure to define: model, test_loader, loss_func, device
# avg_test_loss, avg_test_acc = test(model, test_loader, loss_func, device)