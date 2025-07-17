import torch
import os
import pandas as pd

def test(model, test_loader, loss_func, device, yaml_filename, base_results_dir='results', save_results=False, epoch=None):
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

    if save_results:
        yaml_base = os.path.splitext(os.path.basename(yaml_filename))[0]
        result_dir = os.path.join(base_results_dir, yaml_base)
        os.makedirs(result_dir, exist_ok=True)

        test_filename = os.path.join(result_dir, 'test_results_per_epoch.csv')

        test_data = {
            'Epoch': epoch if epoch is not None else 'N/A',
            'YAML_File': yaml_filename,
            'Test_Loss': avg_test_loss,
            'Test_Accuracy': avg_test_acc
        }

        # Append to CSV
        write_header = not os.path.exists(test_filename)
        df = pd.DataFrame([test_data])
        df.to_csv(test_filename, mode='a', header=write_header, index=False)

        print(f"Test results for epoch {epoch if epoch is not None else 'N/A'} saved to {test_filename}")

    return avg_test_loss, avg_test_acc
