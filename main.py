import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10

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
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss/len(train_set):.3f}.. ")
            
    torch.save(model.state_dict(), 'checkpoint.pth')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    
    model = MyAwesomeModel()

    state_dict = torch.load(model_checkpoint)
    # print("Our model: \n\n", model, '\n')
    # print("The state dict keys: \n\n", state_dict.keys())

    model.load_state_dict(state_dict)

    _, test_set = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_set:
            output = model(images)
            test_loss += criterion(output, labels)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        else:
            print(f"Test loss: {test_loss/len(test_set):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_set):.3f}.. ")
            


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()