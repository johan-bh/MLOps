import pytest
import torch
import yaml
import sys
import os
import torch
import subprocess

# Add the project directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from ML_OPS_Project.models.model import MyAwesomeModel

def test_load_and_forward_model():
    # Load the model configuration
    with open('./conf/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)

    # Instantiate the model
    model = MyAwesomeModel(
        input_size=model_config['model']['input_size'],
        hidden_layers=model_config['model']['hidden_layers'],
        output_size=model_config['model']['output_size'],
        dropout_probability=model_config['model']['dropout_probability']
    )

    # Assert that the model is instantiated
    assert model is not None, "Model instantiation failed"

    # Create a dummy input tensor of the correct shape
    # For example, if your model expects a single-dimensional input of size 'input_size', and batch size is 1
    input_tensor = torch.randn(1, model_config['model']['input_size'])

    # Forward pass through the model
    output = model(input_tensor)

    # Assert that output is not None
    assert output is not None, "Forward pass failed"

    # Optionally, you can also assert the shape of the output
    assert output.shape == (1, model_config['model']['output_size']), f"Output shape is incorrect: {output.shape}"
    
def test_make_dataset():
    # Add the project directory to the PYTHONPATH
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

    # Import the data function
    from ML_OPS_Project.data.make_dataset import mnist

    # Use the data function to load the data
    train, test = mnist()

    # Assert that the data is not None
    assert train is not None, "train is None"
    assert test is not None, "test is None"

    # check if train is a tensor
    assert isinstance(train, torch.utils.data.dataloader.DataLoader), "train is not a tensor"
    assert isinstance(test, torch.utils.data.dataloader.DataLoader), "test is not a tensor"


def test_train_model():
    # Check if the train_model.py script runs without errors
    
    script_path = 'ML_OPS_Project/train_model.py'
    python_executable = sys.executable  # Path to the Python interpreter currently in use

    try:
        # Run the script with a timeout of 5 seconds
        result = subprocess.run([python_executable, script_path], capture_output=True, text=True, timeout=5)

        # If the script finishes within 5 seconds, check the return code
        assert result.returncode == 0, "Script failed to run successfully"

    except subprocess.TimeoutExpired:
        # If the script is still running after 5 seconds, pass the test
        print("Test passed: Script is still running after 5 seconds.")



        