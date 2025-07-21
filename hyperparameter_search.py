import pyhopper as hp
import torch
import torch.nn as nn
import os
import json
from datetime import datetime
import pytz

from utils import get_train_val_loaders
from train import train 

def run_hyperparameter_search(
    model_id,
    number_channels,
    number_classes,
    image_height, 
    image_width,  
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
    base_results_dir
):

    def objective(params):
        from models import model_0, model_1, model_2, model_3, model_4, model_5
        from baseline_models import baseline_model_1, baseline_model_2, baseline_model_3, baseline_model_4, baseline_model_5

        current_batch_size = params["batch_size"]
        current_learning_rate = params["lr"]

        print(f"\n--- Hyperparameter Trial ---")
        print(f"Batch Size: {current_batch_size}, Learning Rate: {current_learning_rate}")

        if model_id == 1:
            model = model_1(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 2:
            model = model_2(
                number_channels, out_channels, image_height, image_width,
                fc_dropout_rate,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 3:
            model = model_3(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )

        elif model_id == 0:
            model = model_0(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )

        elif model_id == 4:
            model = model_4(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        elif model_id == 5:
            model = model_5(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        elif model_id == 6:
            model = baseline_model_1(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 7:
            model = baseline_model_2(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 8:
            model = baseline_model_3(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate,
            )
        elif model_id == 9:
            model = baseline_model_4(
                in_channels=number_channels,
                out_channels=out_channels,
                image_height=image_height,
                image_width=image_width,
                fc_hidden_size=fc_hidden_size,
                number_classes=number_classes,
                fc_dropout_rate=fc_dropout_rate,
                num_layers=num_layers,
            )
        elif model_id == 10:
            model = baseline_model_5(
                number_channels, out_channels, image_height, image_width,
                fc_hidden_size, number_classes, fc_dropout_rate, num_layers,
            )
        else:
            raise ValueError(f"Invalid model ID: {model_id}")

        model = model.to(device)

        current_optimizer = torch.optim.Adam(model.parameters(), lr=current_learning_rate)

        train_loader, val_loader = get_train_val_loaders(
            train_dataset,
            batch_size=current_batch_size,
            val_ratio=val_ratio,
            num_workers=os.cpu_count(),
            seed=seed
        )


        last_epoch_metrics = train(
            model,
            train_loader,
            current_optimizer,
            loss_func,
            epochs,
            device,
            val_loader=val_loader,
            config_filename=config_filename,
            save_outputs=False 

        )

        return last_epoch_metrics['Val_Loss']

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
    print(f"Type of results object: {type(results)}")
    print(f"Directory of results object: {dir(results)}")


    print("Hyperparameter Search Completed.")

    

    print("Hyperparameter Search Completed.")


    if hasattr(results, 'best_f') and results.best_f is not None:
        best_params = results.best_params
        best_loss = results.best_f

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
        # Optionally, print the results object itself to debug:
        print(f"Debug: Results object: {results}")