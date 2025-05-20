import torch

def test(model, test_loader, loss_func, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = loss_func(output, target)
            test_loss += loss.item()
            _, pred = output.max(1)
            test_acc += target.eq(pred).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = 100. * test_acc / len(test_loader.dataset)
    print(f'Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_acc}%')

    return avg_test_loss, avg_test_acc

# Sample usage
# avg_test_loss, avg_test_acc = test(model, test_loader, loss_func, device)
# print(f'Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_acc}%')
