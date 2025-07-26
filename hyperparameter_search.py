import pyhopper as hp
import torch
import torch.nn as nn
import os
import json
from datetime import datetime
import pytz
from utils import get_train_val_loaders
from models import model_0, model_1, model_2, model_3, model_4, model_5
from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5


##########################################################################TRAINING 
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
        return 0.0
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
    clip_coef = (clip_factor * total_norm) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm.item()

def train_for_hyperparam_search(model, train_loader, optimizer, loss_func, epochs, device, val_loader=None):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Optionally print or log the epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader)}")

    # Evaluate on validation set
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss


##########################################################################PYHOPPER 

def run_hyperparameter_search(
    model_id,
    number_channels,
    number_classes,
    image_height, 
    image_width,
    in_channels,  
    out_channels,
    fc_hidden_size,
    fc_dropout_rate,
    num_layers,
    epochs,
    val_ratio,
    seed,
    device,
    loss_func,
    train_dataset,
    config_filename,
    base_results_dir,
    kernel_size,
    stride,
    padding,
    dropout_rate,
    learning_rate
):
    import pyhopper as hp
    import os
    import torch
    from utils import get_train_val_loaders
    from models import model_0, model_1, model_2, model_3, model_4, model_5
    from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5

    def objective(params):
        current_batch_size = params["batch_size"]
        current_learning_rate = params["lr"]

        print(f"\n--- Hyperparameter Trial ---")
        print(f"Batch Size: {current_batch_size}, Learning Rate: {current_learning_rate}")

        if model_id == 1:
            model = model_1(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 2:
            model = model_2(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 3:
            model = model_3(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 0:
            model = model_0(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 4:
            model = model_4(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,

            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 5:
            model = model_5(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 6:
            model = baseline_model_1(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 7:
            model = baseline_model_2(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 8:
            model = baseline_model_3(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            # fc_input_size,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 9:
            model = baseline_model_4(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        elif model_id == 10:
            model = baseline_model_5(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout_rate,
            image_height,
            image_width,
            fc_hidden_size,
            number_classes,
            fc_dropout_rate,
            num_layers,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

        else:
            print("Please input a valid model ID")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate, weight_decay=1e-5)

        train_loader, val_loader = get_train_val_loaders(
            train_dataset,
            batch_size=current_batch_size,
            val_ratio=val_ratio,
            num_workers=os.cpu_count(),
            seed=seed
        )

        val_loss = train_for_hyperparam_search(
            model,
            train_loader,
            optimizer,
            loss_func,
            epochs,
            device,
            val_loader=val_loader
        )

        return val_loss

    search_space = hp.Search({
          "batch_size": hp.int(32, 128, power_of=2),
          "lr": hp.float(1e-5, 1e-2, "0.1g"),

    })

    print("Starting Hyperparameter Search...")
    results = search_space.run(
        objective,
        "minimize",
        "4h",
        n_jobs=1
    )
    print("Hyperparameter Search Completed.")

    if hasattr(results, 'best_f') and results.best_f is not None:
        best_params = results.best_params
        best_loss = results.best_f

        from datetime import datetime
        import pytz
        import json

        sa_timezone = pytz.timezone('Africa/Johannesburg')
        current_time_sast = datetime.now(sa_timezone)
        current_timestamp = current_time_sast.strftime("%Y%m%d_%H%M%S")

        model_dir_name = f"model_{model_id}"
        optimization_results_dir = os.path.join(base_results_dir, "pyhopper", model_dir_name)
        os.makedirs(optimization_results_dir, exist_ok=True)

        optimization_filename = os.path.join(
            optimization_results_dir,
            f"model{model_id}_optimization.json"
        )

        data_to_save = {
            "model_id": model_id,
            "config_file_used_for_search": config_filename,
            "optimal_parameters": best_params,
            "best_validation_loss": best_loss
        }

        try:
            with open(optimization_filename, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Optimal parameters for Model {model_id} saved to: {optimization_filename}")
        except Exception as e:
            print(f"Error saving optimal parameters for Model {model_id}: {e}")
    else:
        print("No optimal parameters found or search did not return expected results.")
        print(f"Debug: Results object: {results}")
