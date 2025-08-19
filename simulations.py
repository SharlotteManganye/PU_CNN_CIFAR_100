import os
import torch
import pandas as pd
from datetime import datetime
import pytz


from train import train  
from test import test     


def run_simulations(model_class,
                    train_loader,
                    test_loader,
                    optimizer,
                    loss_func,
                    num_runs,
                    epochs,
                    batch_size,
                    lr,
                    seed,
                    device,
                    config_filename,
                    results_dir="results/simulations"):

    os.makedirs(results_dir, exist_ok=True)

    for run in range(1, num_runs + 1):
        print(f"\n=== Simulation {run} ===")

        # Handle class vs instance
        if isinstance(model_class, type):
            model = model_class().to(device)
        else:
            model = model_class.to(device)

        model_name = os.path.splitext(os.path.basename(config_filename))[0]
        sim_dir = os.path.join(results_dir, model_name, f"simulation_{run}")
        os.makedirs(sim_dir, exist_ok=True) 



        epoch_metrics = []
        for epoch in range(1, epochs + 1):
            # Train one epoch
            epoch_result = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_func=loss_func,
                epochs=1,   # train for one epoch at a time
                device=device,
                config_filename=config_filename,
                base_results_dir=results_dir,
                save_outputs=False
            )

            train_loss = epoch_result[0]["Train_Loss"]
            train_acc  = epoch_result[0]["Train_Accuracy"]

            # Evaluate on test set
            test_loss, test_acc = test(
                model,
                test_loader,
                loss_func,
                device,
                config_filename,
                base_results_dir=results_dir
            )

            # Save metrics
            metrics = {
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Train_Accuracy": train_acc,
                "Test_Loss": test_loss,
                "Test_Accuracy": test_acc
            }
            epoch_metrics.append(metrics)

            print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
          



        csv_filename = "epoch_metrics.csv"
        csv_path = os.path.join(sim_dir, csv_filename)
        df = pd.DataFrame(epoch_metrics)
        df.to_csv(csv_path, index=False)
        print(f"Metrics for run {run} saved to {csv_path}")
