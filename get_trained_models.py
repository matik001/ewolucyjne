import glob
import os
import torch
import wandb

from data_loaders import get_mnist_loaders, get_cifar10_loaders, get_svhn_loaders
from training_utils import train_model, eval_model

def get_newest_files(directory, num_files=9):
    files = glob.glob(os.path.join(directory, '*'))
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    newest_files = sorted_files[:num_files]
    return newest_files

def train_saved_models(device = "cuda"):
    epochs = 20
    learning_rate = 0.001
    model_files = get_newest_files("saved_models")
    for model_file in model_files:
        print(f"Training model: {model_file}")
        model_file_cpy = model_file
        model_file = model_file.split("/")[1]
        dataset = model_file.split('_')[0]
        model_name = model_file.split("\\")[-1].split(".")[0]
        phase = model_name.split("_")[1]
        model = torch.load(model_file_cpy).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader, val_loader, test_loader = None, None, None
        if "MNIST" in model_file:
            print("Using MNIST dataset")
            train_loader, val_loader, test_loader = get_mnist_loaders(False)
        elif "CIFAR10" in model_file:
            print("Using CIFAR-10 dataset")
            train_loader, val_loader, test_loader = get_cifar10_loaders(False)
        elif "SVHN" in model_file:
            print("Using SVHN dataset")
            train_loader, val_loader, test_loader = get_svhn_loaders(False)
        run = wandb.init(project="NAS Hyperparameter Tuning", name=model_name, tags=[dataset, "baseline", phase])
        run.log({
            "epochs" : epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "adam",
            "learning_rate": 0.001,  
            "architecture": [layer for _, layer in model.named_children()],
            f"total_parameters": sum(p.numel() for p in model.parameters()),
            f"num_layers": len(model)
        })
        train_model(model, train_loader, epochs, run, optimizer, device)
        eval_model(model, val_loader, run, device, "Validation")
        eval_model(model, test_loader, run, device, "Test")
        run.finish()

if __name__ == "__main__":
    train_saved_models("cuda")