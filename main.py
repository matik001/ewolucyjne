import wandb

from optimizer.NASOptimizer import NASOptimizer
from data_loaders import get_mnist_loaders, get_mnist_loaders_prev


def run_optimizer(device="cuda"):
    train_loader, test_loader = get_mnist_loaders_prev(True)
    # train_loader, val_loader, test_loader = get_mnist_loaders(True, 0.1)

    nas = NASOptimizer(
        input_shape=(1, 28, 28),  # MNIST format
        num_classes=10,
        population_size=10,
        num_generations=5,
        mutation_rate=0.3,
        elite_size=2,
        project_name="MNIST-NAS",
        epoch=5
    )

    best_network = nas.optimize(train_loader, test_loader, device)
    print("\nNajlepsza znaleziona architektura:")
    print(best_network)

def main(device="cuda"):
    run_optimizer(device)


if __name__ == "__main__":
    wandb.login()

    main()
    # deepseek()
    # claude()
    # chatgpt()