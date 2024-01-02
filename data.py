import torch
from torch.utils.data import TensorDataset, DataLoader

def mnist(batch_size=32):
    """Return train and test dataloaders for MNIST."""
    # Path to the folder containing .pt files
    path = "../../../data/corruptmnist/"

    # Load training data
    train_images = torch.cat([torch.load(f"{path}train_images_{i}.pt") for i in range(6)])
    train_targets = torch.cat([torch.load(f"{path}train_target_{i}.pt") for i in range(6)])
    train_dataset = TensorDataset(train_images, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test data
    test_images = torch.load(f"{path}test_images.pt")
    test_targets = torch.load(f"{path}test_target.pt")
    test_dataset = TensorDataset(test_images, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader