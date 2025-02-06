from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_mnist_loaders_prev(limit = False):
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
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader