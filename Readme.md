# Plant Seedling Classification
Welcome to the Plant Seedling Classification repository! This codebase contains the necessary code to train and evaluate a model for classifying weed and crop seedlings.

## Data
The dataset used in this project consists of images of approximately 960 unique plants belonging to 12 species at various growth stages. It has been provided by the Aarhus University Signal Processing group in collaboration with the University of Southern Denmark.

## Code Overview
The following files are included in this repository:

- `preprocess.py`: This script contains functions for preprocessing the raw image data and creating train and test datasets.
- `train_steps.py`: This script contains functions for training and evaluating a model on the dataset.
- `train_model.py`: This script is the entry point for training and evaluating a model. It uses the functions from the other two scripts to preprocess the data, train a model, and evaluate it on the test set.
- `Plant_seed_classification.ipynb`: This script contains we plant seed classification model evaluation from top to end
Running the Code
To run the code, you will need to install the necessary dependencies and download the data. Once these steps are complete, you can run the main.py script to train and evaluate the model.

## Dependencies
This codebase requires the following Python packages:

- numpy
- pandas
- torch
- torchvision
-
## Acknowledgments

I would like to extend my appreciation to the Aarhus University Department of Engineering Signal Processing Group for hosting the original data.
