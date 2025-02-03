import torch
import torchvision
import wandb
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from optimizer.Chromosome import Chromosome
from optimizer.Layers import Conv2dLayer, ReluLayer, MaxPool2dLayer, DropoutLayer, LinearLayer
from optimizer.NASOptimizer import NASOptimizer


def get_mnist_loaders(limit = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # if limit:
    #     train_dataset = Subset(train_dataset, list(range(64*1)))
    #     test_dataset = Subset(test_dataset, list(range(64*1)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader




def crossover_example1():
    chrom1 = Chromosome((3, 32, 32), 10)
    chrom1.add_layer(Conv2dLayer((3, 32, 32), 16, 3))
    chrom1.add_layer(ReluLayer((16, 30, 30)))
    chrom1.add_layer(Conv2dLayer((16, 30, 30), 32, 3))
    chrom1.add_layer(MaxPool2dLayer((32, 28, 28), 2))
    print(chrom1)

    chrom2 = Chromosome((3, 32, 32), 10)
    chrom2.add_layer(Conv2dLayer((3, 32, 32), 24, 3))
    chrom2.add_layer(MaxPool2dLayer((24, 30, 30), 2))
    chrom2.add_layer(ReluLayer((24, 15, 15)))
    print(chrom2)

    child = chrom1.crossover(chrom2)
    print(child)



def crossover_example2():
    chrom1 = Chromosome((3, 224, 224))

    chrom1.add_layer(Conv2dLayer((3, 224, 224), 64, 3, padding=1))
    chrom1.add_layer(ReluLayer(chrom1.get_output_shape()))
    chrom1.add_layer(Conv2dLayer(chrom1.get_output_shape(), 64, 3, padding=1))
    chrom1.add_layer(ReluLayer(chrom1.get_output_shape()))
    chrom1.add_layer(MaxPool2dLayer(chrom1.get_output_shape(), 2))

    chrom1.add_layer(Conv2dLayer(chrom1.get_output_shape(), 128, 3, padding=1))
    chrom1.add_layer(ReluLayer(chrom1.get_output_shape()))
    chrom1.add_layer(Conv2dLayer(chrom1.get_output_shape(), 128, 3, padding=1))
    chrom1.add_layer(ReluLayer(chrom1.get_output_shape()))
    chrom1.add_layer(MaxPool2dLayer(chrom1.get_output_shape(), 2))

    chrom1.add_layer(Conv2dLayer(chrom1.get_output_shape(), 256, 3, padding=1))
    chrom1.add_layer(ReluLayer(chrom1.get_output_shape()))
    chrom1.add_layer(DropoutLayer(chrom1.get_output_shape(), 0.3))
    chrom1.add_layer(MaxPool2dLayer(chrom1.get_output_shape(), 2))

    chrom1.add_layer(LinearLayer(chrom1.get_output_shape(), 512))
    chrom1.add_layer(ReluLayer((512,)))
    chrom1.add_layer(DropoutLayer((512,), 0.5))
    chrom1.add_layer(LinearLayer((512,), 10))  # 10 klas wyjściowych

    chrom2 = Chromosome((3, 224, 224))

    chrom2.add_layer(Conv2dLayer((3, 224, 224), 32, 5, padding=2))
    chrom2.add_layer(ReluLayer(chrom2.get_output_shape()))
    chrom2.add_layer(MaxPool2dLayer(chrom2.get_output_shape(), 2))

    chrom2.add_layer(Conv2dLayer(chrom2.get_output_shape(), 64, 5, padding=2))
    chrom2.add_layer(ReluLayer(chrom2.get_output_shape()))
    chrom2.add_layer(Conv2dLayer(chrom2.get_output_shape(), 64, 3, padding=1))
    chrom2.add_layer(ReluLayer(chrom2.get_output_shape()))
    chrom2.add_layer(MaxPool2dLayer(chrom2.get_output_shape(), 2))

    chrom2.add_layer(Conv2dLayer(chrom2.get_output_shape(), 128, 3, padding=1))
    chrom2.add_layer(ReluLayer(chrom2.get_output_shape()))
    chrom2.add_layer(DropoutLayer(chrom2.get_output_shape(), 0.4))

    chrom2.add_layer(LinearLayer(chrom2.get_output_shape(), 256))
    chrom2.add_layer(ReluLayer((256,)))
    chrom2.add_layer(LinearLayer((256,), 10))

    print("Chromosom 1:")
    print(chrom1)
    print("\nChromosom 2:")
    print(chrom2)

    # Test mutacji
    print("\nPo mutacji chromosomu 1:")
    mutated = chrom1.mutate()
    print(mutated)

    # Test crossover
    print("\nPo crossover:")
    child = chrom1.crossover(chrom2)
    print(child)


def create_mnist_chromosome():
    # Tworzymy chromosom dla MNIST (input: 1x28x28)
    chrom = Chromosome(input_shape=(1, 28, 28), num_classes=10)

    # Layer 1
    chrom.add_layer(Conv2dLayer((1, 28, 28), 32, kernel_size=5, stride=1, padding=2))
    chrom.add_layer(ReluLayer(chrom.get_output_shape()))
    chrom.add_layer(MaxPool2dLayer(chrom.get_output_shape(), kernel_size=2, stride=2))

    # Layer 2
    chrom.add_layer(Conv2dLayer(chrom.get_output_shape(), 64, kernel_size=5, stride=1, padding=2))
    chrom.add_layer(ReluLayer(chrom.get_output_shape()))
    chrom.add_layer(MaxPool2dLayer(chrom.get_output_shape(), kernel_size=2, stride=2))

    # Fully connected layers
    chrom.add_layer(DropoutLayer(chrom.get_output_shape()))
    chrom.add_layer(LinearLayer(chrom.get_output_shape(), 1000))
    chrom.add_layer(LinearLayer((1000,), 10))
    is_correct, error = chrom.is_correct()
    print("Creating chromosome")
    print("Is correct: ", is_correct, error)
    if not is_correct:
        exit(1)
    return chrom


def test_mnist_training():
    train_loader, test_loader = get_mnist_loaders()
    # chrom = create_mnist_chromosome()
    chrom = Chromosome.generate_random((1, 28, 28), 10)

    # Sprawdź poprawność architektury
    is_valid, message = chrom.is_correct()
    print(f"Sprawdzenie architektury:")
    print(f"Poprawna: {is_valid}")
    print(f"Komunikat: {message}")
    print("\nArchitektura sieci:")
    print(chrom)

    # Parametry treningu
    params = {
        'learning_rate': 0.001,
        'num_epochs': 2,  # Zmniejszona liczba epok dla szybszego testu
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\nRozpoczynam trening na urządzeniu: {params['device']}")

    # Oblicz fitness
    fitness = chrom.get_fitness(
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=params['learning_rate'],
        num_epochs=params['num_epochs'],
        device=params['device']
    )

    print(f"\nKońcowy wynik fitness: {fitness}")

    return chrom, fitness

def run_optimizer():
    train_loader, test_loader = get_mnist_loaders(True)

    nas = NASOptimizer(
        input_shape=(1, 28, 28),  # MNIST format
        num_classes=10,
        population_size=10,
        num_generations=5,
        mutation_rate=0.3,
        elite_size=2,
        project_name="MNIST-NAS"
    )

    best_network = nas.optimize(train_loader, test_loader)
    print("\nNajlepsza znaleziona architektura:")
    print(best_network)

def main():
    # crossover_example2()
    # test_mnist_training()
    run_optimizer()


if __name__ == "__main__":
    wandb.login()

    main()
    # deepseek()
    # claude()
    # chatgpt()