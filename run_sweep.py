import wandb
import torch

from data_loaders import get_mnist_loaders, get_cifar10_loaders, get_svhn_loaders
from training_utils import train_model, eval_model
from get_trained_models import get_newest_files

data_loaders = {}


def main(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config 
        model_name = config.model_path.split("\\")[-1].split(".")[0]
        phase = model_name.split("_")[1]
        dataset = model_name.split("_")[0]
        run.name = f"{model_name}_{run.id[:4]}"
        run.tags = ["tuning", dataset, phase]
        run.save()

        train_loader, val_loader, test_loader = data_loaders[f"{dataset}_{batch_size}"]
        # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=config.batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(val_loader, batch_size=config.batch_size, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(test_loader, batch_size=config.batch_size, shuffle=False)

        model = torch.load(config.model_path).to(config.device)
        run.log({
            "architecture": [layer for _, layer in model.named_children()],
            f"total_parameters": sum(p.numel() for p in model.parameters()),
            f"num_layers": len(model)
        })

        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

        train_model(model, train_loader, config.epochs, run, optimizer, device=config.device)
        eval_model(model, val_loader, run, config.device, "Validation")
        eval_model(model, test_loader, run, config.device, "Test")


batch_sizes = [32, 64, 256]
datasets = ["MNIST", "CIFAR10", "SVHN"]
sweep_config = {
    'method': 'grid',
    'name': 'sweep - NAS Hyperparameter Tuning',
    'metric': {'goal': 'maximize', 'name': 'Validation/accuracy'},
    'parameters':
    {
        'model_path' : {'values': get_newest_files("saved_models")},
        'batch_size': {'values': batch_sizes},
        'optimizer': {'values': ['adam', 'sgd']},
        'learning_rate': {'values': [1e-3, 1e-2]},  
        'weight_decay': {'values': [1e-4, 1e-3]},  
        'momentum': {'values': [0.9, 0.99]},  
        'epochs': {'values': [20]},
        'device': {'values': ["cuda"]},
        },
}

for dataset in datasets:
    for batch_size in batch_sizes:
        if dataset == "MNIST":
            train_loader, val_loader, test_loader = get_mnist_loaders(False, batch_size=batch_size)
        elif dataset == "CIFAR10":
            train_loader, val_loader, test_loader = get_cifar10_loaders(False, batch_size=batch_size)
        elif dataset == "SVHN":
            train_loader, val_loader, test_loader = get_svhn_loaders(False, batch_size=batch_size)
        data_loaders[f"{dataset}_{batch_size}"] = (train_loader, val_loader, test_loader)


sweep_id = wandb.sweep(sweep_config, project="NAS Hyperparameter Tuning")
wandb.agent(sweep_id, main, count=5)
wandb.finish()