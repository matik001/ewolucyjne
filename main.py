import wandb
import torch
from torch import nn, optim

from optimizer.NASOptimizer import NASOptimizer
from data_loaders import get_mnist_loaders, get_mnist_loaders_prev


def run_optimizer(device="cuda"):
    train_loader, val_loader, test_loader = get_mnist_loaders(False)

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
    # nas = NASOptimizer(
    #     input_shape=(1, 28, 28),  # MNIST format
    #     num_classes=10,
    #     population_size=2,
    #     num_generations=1,
    #     mutation_rate=0.3,
    #     elite_size=2,
    #     project_name="MNIST-NAS",
    #     epoch=3
    # )

    best_network = nas.optimize(train_loader, val_loader, test_loader, device)
    print("\nNajlepsza znaleziona architektura:")
    print(best_network)
    model = best_network.to_nn_module().to(device)
    model.eval()
    criterion = nn.NLLLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    # run.log({
    #     f"test/loss": test_loss,
    #     f"test/accuracy": test_acc,
    # })
    print(f"Test, Test Loss: {test_loss}, Test Accuracy: {test_acc}%")

def main(device="cuda"):
    run_optimizer(device)


if __name__ == "__main__":
    wandb.login()

    main()
    # deepseek()
    # claude()
    # chatgpt()