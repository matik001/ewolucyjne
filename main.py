import wandb
import torch
from torch import nn, optim

from optimizer.NASOptimizer import NASOptimizer
from data_loaders import get_mnist_loaders, get_mnist_loaders_prev
from training_utils import eval_model, train_model

# import sys

# # Redirecting stdout to a file
# sys.stdout = open('output.txt', 'w')


def run_optimizer(device="cuda"):
    train_loader, val_loader, test_loader = get_mnist_loaders(False)
    # train_loader, test_loader, val_loader = get_mnist_loaders(False)

    # nas = NASOptimizer(
    #     input_shape=(1, 28, 28),  # MNIST format
    #     num_classes=10,
    #     population_size=10,
    #     num_generations=5,
    #     mutation_rate=0.3,
    #     elite_size=2,
    #     project_name="MNIST-NAS",
    #     epoch=5
    # )
    print(len(test_loader))
    epoch = 2
    nas = NASOptimizer(
        input_shape=(1, 28, 28),  # MNIST format
        num_classes=10,
        population_size=1,
        num_generations=3,
        mutation_rate=0.3,
        elite_size=2,
        project_name="MNIST-NAS",
        epoch=2,
        min_layers=epoch,
        max_layers=4
    )

    best_network = nas.optimize(train_loader, val_loader, test_loader, device)
    print("\nNajlepsza znaleziona architektura:")
    print(best_network)
    model = best_network.to_nn_module().to(device)
    train_model(model, train_loader, epoch, device)
    eval_model(model, test_loader, device)

def main(device="cuda"):
    run_optimizer(device)


if __name__ == "__main__":
    wandb.login()

    main("cuda")
