# ML_OPS_Project

This repository was made as part of the exercises of the ML Ops course.
The project uses as ML Ops cookiecutter template and the overall project structure can be seen in the section below. The project consists of a simple fully connected linear neural network that is trained on corrupted mnist data. 

The primary scripts of the project are (note all the commands are run from the root folder "ML_Ops_Project"):

### model.py: 
This python file defines the neural network in pytorch

### make_dataset.py:
This python file creates a dataset / dataloader using the raw pytorch files from the data/raw folder - this script is invoked and used in the train_model.py script. Furthemore, it saves a normalized tensor with train/test data in the data/processed folder

### train_model.py:
This python file run the training of the neural network or evaluation of the neural network based on the commands passed. If the command train is passed it will invoke make_dataset and load the model from model.py. The trained model is saved as pytorch file in models/trained_models with the name "trained_model.pth". Furthermore, a training loss curve will be saved to the reports/figures folder. If the evaluate command is passed you'll need to specify the name of the saved model which will yield the accurary on the test data.

#### Example of commands:
    
    python .\ML_OPS_Project\train_model.py --lr 1e-4

    python .\ML_OPS_Project\train_model.py evaluate trained_model.pth

### predict_model.py:
This python file will load a trained model and make predictions on raw images. To use this functionality simply pass the folder name containing the images.

Example of commmand:

    python .\ML_OPS_Project\predict_model.py trained_model.pth raw_images

### visualize.py
This python file will load a pre-trained network and visualize intermediate representations of the data in 2D space using t-SNE. The result will be saved in reports/figures/feature_visualization.png

Example of command:

    python .\ML_OPS_Project\visualizations\visualize.py
    


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── ML_OPS_Project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
