import random
import wandb

from optimizer.NASOptimizer import NASOptimizer
from data_loaders import get_mnist_loaders, get_cifar10_loaders, get_svhn_loaders
from training_utils import eval_model, train_model, save_model, get_dataloader_info

# import sys

# # Redirecting stdout to a file
# sys.stdout = open('output.txt', 'w')


def run_optimizer(dataset = "MNIST", device="cuda"):
    train_loader, val_loader, test_loader = None, None, None
    if dataset == "MNIST":
        print("Using MNIST dataset")
        train_loader, val_loader, test_loader = get_mnist_loaders(False)
    elif dataset == "CIFAR10":
        print("Using CIFAR-10 dataset")
        train_loader, val_loader, test_loader = get_cifar10_loaders(False)
    elif dataset == "SVHN":
        print("Using SVHN dataset")
        train_loader, val_loader, test_loader = get_svhn_loaders(False)

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
    input_shape, num_classes = get_dataloader_info(test_loader)
    population_size = 10
<<<<<<< HEAD
    num_generations = 3
=======
    num_generations = 7
>>>>>>> e873c050865dbd7de993d21f3899ca967e17b605
    mutation_rate = 0.3
    elite_size = 2
    project_name = "NAS"
    group_name = dataset
    epoch = 20
    min_layers = 3
    max_layers = 10
    nas = NASOptimizer(
        input_shape=input_shape,  
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

    best_chromosome = nas.optimize(train_loader, val_loader, save_initial=True, save_medium=True, device=device)
    print("\nNajlepsza znaleziona architektura:")
    print(best_chromosome)
    model = best_chromosome.to_nn_module().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    chromosome_id = random.randint(1000, 9999)
    run =  wandb.init(project=project_name, group=group_name, name=f"{project_name}_{group_name}_FINAL_{chromosome_id}", tags=[ group_name, "FINAL"])
    run.config.update({
        "input_shape": input_shape,
        "num_classes": num_classes,
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "elite_size": elite_size,
        "generation": num_generations,
        "chromosome_id": chromosome_id,
    })
    run.log({
        "architecture": str(best_chromosome),
        f"total_parameters": sum(p.numel() for p in model.parameters()),
        f"num_layers": len(best_chromosome.layers)
    })
    train_model(model, train_loader, epoch, run, optimizer, device)
    eval_model(model, val_loader, run, device, "Validation")
    eval_model(model, test_loader, run,  device, "Test")
    run.finish()
    save_model(best_chromosome, f"saved_models/{group_name}_best_{chromosome_id}.pth")

def main(dataset="MNIST", device="cuda"):
    run_optimizer(dataset, device)

if __name__ == "__main__":
    wandb.login()
    main("MNIST", "cuda:1")
    main("SVHN", "cuda:1")
    main("CIFAR10", "cuda:1")
