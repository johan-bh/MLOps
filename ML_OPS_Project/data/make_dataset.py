import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def mnist(batch_size=32):
    """Return train and test dataloaders for MNIST."""
    # Path to the folder containing .pt files
    current_script_path = os.path.abspath(__file__)

    # Get the directory of the 'make_dataset.py' file
    current_script_dir = os.path.dirname(current_script_path)

    # Construct the absolute path to the 'raw' data folder
    raw_data_path = os.path.join(current_script_dir, "..", "..", "data", "raw")
    processed_data_path = os.path.join(current_script_dir, "..", "..", "data", "processed")

    # Load training data
    train_images = torch.cat([torch.load(os.path.join(raw_data_path, f"train_images_{i}.pt")) for i in range(10)])
    train_targets = torch.cat([torch.load(os.path.join(raw_data_path, f"train_target_{i}.pt")) for i in range(10)])
    train_dataset = TensorDataset(train_images, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test data
    # Load test data using the absolute path
    test_images = torch.load(os.path.join(raw_data_path, "test_images.pt"))
    test_targets = torch.load(os.path.join(raw_data_path, "test_target.pt"))
    test_dataset = TensorDataset(test_images, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Save intermediate datasets to processed data folder
    train_mean, train_std = train_images.mean(), train_images.std()
    test_mean, test_std = test_images.mean(), test_images.std()

    # Normalize and save the datasets
    normalize_and_save(train_images, train_mean, train_std, os.path.join(processed_data_path, "normalized_train.pt"))
    normalize_and_save(test_images, test_mean, test_std, os.path.join(processed_data_path, "normalized_test.pt"))

    return train_loader, test_loader


def normalize_and_save(data, mean, std, file_path):
    # Normalize the data
    normalized_data = (data - mean) / std

    # Save the normalized data
    torch.save(normalized_data, file_path)


mnist()
