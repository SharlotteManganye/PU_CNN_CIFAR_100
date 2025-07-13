import torch
import os
import pandas as pd
from datetime import datetime
import pytz

def test(model, test_loader, loss_func, device, yaml_filename, base_results_dir='results'):
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

    print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")

    yaml_base = os.path.splitext(os.path.basename(yaml_filename))[0]
    result_dir = os.path.join(base_results_dir, yaml_base)
    os.makedirs(result_dir, exist_ok=True)

    sa_timezone = pytz.timezone('Africa/Johannesburg')
    timestamp = datetime.now(sa_timezone).strftime("%Y%m%d_%H%M%S")
    test_filename = os.path.join(result_dir, f'test_results_{timestamp}.csv')

    test_data = {
        'Timestamp': timestamp,
        'YAML_File': yaml_filename,
        'Test_Loss': avg_test_loss,
        'Test_Accuracy': avg_test_acc
    }
    pd.DataFrame([test_data]).to_csv(test_filename, index=False)

    print(f"Test results saved to {test_filename}")
    return avg_test_loss, avg_test_acc
