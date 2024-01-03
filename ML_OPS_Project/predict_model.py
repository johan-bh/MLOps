import os
import torch
import numpy as np
from models.model import MyAwesomeModel
from torchvision import transforms
from PIL import Image
import click

# Path to the folder containing .pt files
current_script_path = os.path.abspath(__file__)

# Get the directory of the 'make_dataset.py' file
current_script_dir = os.path.dirname(current_script_path)

# Construct the absolute path to the 'trained_models' folder
trained_models_path = os.path.join(current_script_dir, "models", "trained_models")


# Define a function to load images from a folder
def load_images_from_folder(folder):
    # Construct the absolute path to the raw images folder
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("L")  # convert image to grayscale
        if img is not None:
            images.append(img)
    return images


# Define a function to load images from a numpy/pickle file
def load_images_from_file(file_path):
    return np.load(file_path, allow_pickle=True)


# Define a function to prepare the data for the model
def prepare_data(images, transform):
    tensor_images = torch.stack([transform(image) for image in images])
    return tensor_images


@click.command()
@click.argument("model_name")
@click.argument("folder_name")
def predict(model_name, folder_name):
    """Load a pre-trained model and make predictions on provided data.

    Inputs:
    model_name: Name of the pre-trained model to load
    folder_name: Name of the folder containing the images to predict

    Outputs:
    Prints the predicted labels for each image
    """
    # Load the pre-trained model
    model = MyAwesomeModel()

    # Construct the absolute path to the pre-trained model
    model_path = os.path.join(trained_models_path, model_name)

    # Load the pre-trained model
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # assuming MNIST images
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # assuming MNIST normalization
        ]
    )

    data_path = os.path.join(current_script_dir, "..", "data", f"{folder_name}")

    # Check if data_path is a directory or a file
    if os.path.isdir(data_path):
        # Load images from the folder
        images = load_images_from_folder(data_path)
    elif os.path.isfile(data_path):
        # Load images from the numpy/pickle file
        images = load_images_from_file(data_path)
    else:
        raise ValueError("The provided data path is neither a directory nor a file.")

    # Prepare data for prediction
    input_data = prepare_data(images, transform)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)

    # Print predictions
    for i, pred in enumerate(predicted):
        print(f"Image {i+1}: Predicted Label {pred.item()}")


if __name__ == "__main__":
    predict()
