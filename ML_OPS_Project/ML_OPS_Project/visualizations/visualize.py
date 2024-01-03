import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path
parent_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, parent_dir)

from models.model import MyAwesomeModel
from data.make_dataset import mnist
import os

def load_model(model_path):
    """Load a pre-trained model
    
    Inputs:
    model_path: Path to the pre-trained model
    
    Outputs:
    model: The pre-trained model
    """
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def extract_features(model, dataloader):
    """Extract features from a pre-trained model.
    
    Inputs:
    model: A pre-trained model
    dataloader: A PyTorch dataloader
    
    Outputs:
    features: Extracted features from the model
    labels: Labels corresponding to the extracted features"""
    features = []
    labels = []
    for data, target in dataloader:
        # Forward pass to get features
        data = data.view(data.shape[0], -1)  # Flatten the data
        with torch.no_grad():
            output = model(data)
        
        features.append(output)
        labels.append(target)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels

def visualize_tsne(features, labels, save_path):
    """Visualize extracted features using t-SNE.
    
    Inputs:
    features: Extracted features from the model
    labels: Labels corresponding to the extracted features
    save_path: Path to save the visualization
    """

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of extracted features")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Construct the path for saving the visualization
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    # Construct the absolute path to the 'trained_models' folder
    model_path = os.path.join(current_script_dir, '..','models', 'trained_models')
    model_checkpoint_path = os.path.join(model_path, 'trained_model.pth')
    model = load_model(model_checkpoint_path)
    
    # Assuming mnist function returns dataloaders
    train_loader, _ = mnist()

    features, labels = extract_features(model, train_loader)

    # Construct the path for saving the visualization
    reports_figures_path = os.path.join(current_script_dir, '..', '..', 'reports', 'figures')
    visualization_save_path = os.path.join(reports_figures_path, 'feature_visualization.png')

    visualize_tsne(features, labels, visualization_save_path)
