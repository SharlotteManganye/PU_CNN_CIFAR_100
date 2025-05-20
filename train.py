import torch

def adaptive_clip_grad_norm(parameters, clip_factor=0.01, eps=1e-3):
    if not isinstance(parameters, torch.Tensor):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters:
        return

    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
    clip_coef = (clip_factor * total_norm) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))

def train(model, train_loader, optimizer, loss_func, epochs, device):
    model.train()
    train_loss = 0
    for epoch in range(epochs):
        train_acc = 0
        for i, (data, target) in enumerate(train_loader):  # Added enumerate
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = loss_func(output, target)
            loss.backward()

            adaptive_clip_grad_norm(model.parameters())

            optimizer.step()

            train_loss += loss.item()

            _, pred = output.max(1)
            train_acc += target.eq(pred).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_acc / len(train_loader.dataset)
        print(f'epoch: {epoch+1},Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_acc}%')


    return avg_train_loss, avg_train_acc

def val(model, val_loader, loss_func, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())

            loss = loss_func(output, target)
            val_loss += loss.item()

            _, pred = output.max(1)
            val_acc += target.eq(pred).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100. * val_acc / len(val_loader.dataset)
    # print('\tVal loss: {:.4f}, acc: {:.4f}%'.format(val_loss, val_acc))
    return avg_val_loss, avg_val_acc