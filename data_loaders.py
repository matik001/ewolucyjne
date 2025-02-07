from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_mnist_loaders(limit=False, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Split train dataset into train and validation sets
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Apply limit for debugging
    if limit:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(64)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(64)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(64)))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader

def get_cifar10_loaders(limit=False, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalization for CIFAR-10
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split train dataset into train and validation sets
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Apply limit for debugging
    if limit:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(64)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(64)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(64)))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader

def get_svhn_loaders(limit=False, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # Normalization for SVHN
    ])

    # Load the SVHN datasets
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    # Split the training dataset into train and validation sets
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Apply limit for debugging
    if limit:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(64)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(64)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(64)))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader