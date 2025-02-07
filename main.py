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
    input_shape = (1, 28, 28)
    num_classes = 10
    population_size = 2
    num_generations = 3
    mutation_rate = 0.3
    elite_size = 2
    project_name = "NAS"
    group_name = "MNIST"
    epoch = 2
    min_layers = 2
    max_layers = 4
    nas = NASOptimizer(
        input_shape=input_shape,  # MNIST format
        num_classes=num_classes,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        project_name=project_name,
        group_name=group_name,
        epoch=epoch,
        min_layers=min_layers,
        max_layers=max_layers
    )

    best_chromosome = nas.optimize(train_loader, val_loader, device)
    print("\nNajlepsza znaleziona architektura:")
    print(best_chromosome)
    model = best_chromosome.to_nn_module().to(device)
    run =  wandb.init(project=project_name, group=group_name, name=f"{project_name}_{group_name}_FINAL", tags=[ group_name])
    run.config.update({
        "input_shape": input_shape,
        "num_classes": num_classes,
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "elite_size": elite_size,
        "generation": num_generations,
        "chromosome_id": 777,
    })
    run.log({
        "architecture": str(best_chromosome),
        f"total_parameters": sum(p.numel() for p in model.parameters()),
        f"num_layers": len(best_chromosome.layers)
    })
    train_model(model, train_loader, epoch, run, device)
    eval_model(model, test_loader, run,  device, "Test")
    run.finish()

def main(device="cuda"):
    run_optimizer(device)


if __name__ == "__main__":
    wandb.login()

    main("cuda")
