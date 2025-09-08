import os
import torch
import pandas as pd
from datetime import datetime
import pytz

from train import train
# from test import test

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
                    results_dir="results/simulations",
                    resume_checkpoint_path=None):

    os.makedirs(results_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(config_filename))[0]

    for run in range(1, num_runs + 1):
        print(f"\n=== Simulation {run} ===")

        if isinstance(model_class, type):
            model = model_class().to(device)
        else:
            model = model_class.to(device)

        sim_dir = os.path.join(results_dir, model_name, f"simulation_{run}")
        os.makedirs(sim_dir, exist_ok=True)

        start_epoch = 1
        if resume_checkpoint_path:
            print(f"Resuming simulation from checkpoint: {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {checkpoint['epoch']}")

        all_epoch_metrics = []
        sa_timezone = pytz.timezone('Africa/Johannesburg')

        print(f"Training started for run at {datetime.now(sa_timezone).strftime('%Y%m%d_%H%M%S')} SAST.")
        print(f"Initial Learning Rate: {lr}")
        print(f"Batch Size: {batch_size}")
        
       
        for epoch in range(start_epoch, epochs + 1):
            
           
            train_loss, train_acc = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_func=loss_func,
                device=device,
                epoch=epoch
            )

          
            test_loss, test_acc = test(
                model,
                test_loader,
                loss_func,
                device 
            )

            metrics = {
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Train_Accuracy": train_acc,
                "Test_Loss": test_loss,
                "Test_Accuracy": test_acc
            }
            all_epoch_metrics.append(metrics)

            print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            checkpoint_path = os.path.join(sim_dir, f"{model_name}_checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
          

        csv_filename = "epoch_metrics.csv"
        csv_path = os.path.join(sim_dir, csv_filename)
        df = pd.DataFrame(all_epoch_metrics)
        df.to_csv(csv_path, index=False)
        print(f"Metrics for run {run} saved to {csv_path}")