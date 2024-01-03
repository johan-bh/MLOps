import click
import torch
from models.model import MyAwesomeModel
from data.make_dataset import mnist
import os
import matplotlib.pyplot as plt


# Path to the folder containing .pt files
current_script_path = os.path.abspath(__file__)

# Get the directory of the 'make_dataset.py' file
current_script_dir = os.path.dirname(current_script_path)

# Construct the absolute path to the 'trained_models' folder
trained_models_path = os.path.join(current_script_dir, "models", "trained_models")

# Path to the reports/figures directory
reports_figures_path = os.path.join(current_script_dir, "..", "reports", "figures")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST.

    Inputs:
    lr: learning rate to use for training

    Outputs:
    Saves the trained model in the 'trained_models' folder
    Saves a plot of the training curve in the 'reports/figures' folder
    """
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10

    # Initialize a list to track the loss for each epoch
    epoch_losses = []

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for images, labels in train_set:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        else:
            average_loss = train_loss / len(train_set)
            epoch_losses.append(average_loss)
            print(f"Epoch {epoch+1}/{epochs}.. " f"Train loss: {train_loss/len(train_set):.3f}.. ")

    # Save the model checkpoint
    model_checkpoint_path = os.path.join(trained_models_path, "trained_model.pth")
    torch.save(model.state_dict(), model_checkpoint_path)

    # Generate and save a training curve plot
    plt.figure()
    plt.plot(epoch_losses, label="Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    training_curve_path = os.path.join(reports_figures_path, "training_curve.png")
    plt.savefig(training_curve_path)
    print(f"Training curve saved to {training_curve_path}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model.

    Inputs:
    model_checkpoint: Name of the model checkpoint to load

    Outputs:
    Prints the test loss and accuracy
    """
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here

    model = MyAwesomeModel()

    model_checkpoint_path = os.path.join(trained_models_path, model_checkpoint)
    state_dict = torch.load(model_checkpoint_path)
    # print("Our model: \n\n", model, '\n')
    # print("The state dict keys: \n\n", state_dict.keys())

    # Load the state dict into the network
    model.load_state_dict(state_dict)

    # Load the test set
    _, test_set = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    accuracy = 0

    # Set the model to evaluation mode
    model.eval()
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for images, labels in test_set:
            # Forward pass, then backward pass, then update weights
            output = model(images)
            test_loss += criterion(output, labels)
            # Get the class probabilities
            ps = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = ps.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        else:
            print(f"Test loss: {test_loss/len(test_set):.3f}.. " f"Test accuracy: {accuracy/len(test_set):.3f}.. ")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
