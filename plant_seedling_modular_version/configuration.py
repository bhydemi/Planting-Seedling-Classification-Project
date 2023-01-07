import torch 
from torch import nn 
import os
from models import ResNet
from dataclasses import dataclass
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DataStorage:
    """
    A class for storing and loading data.

    Attributes:
    path_to_save (str): The path where the data should be saved.
    path_to_load (str): The path from where the data should be loaded.
    """
    path_to_save: str = './data/'
    path_to_load: str = "https://www.kaggle.com/competitions/plant-seedlings-classification/data"

class PreprocessConfiguration:
    """
    A configuration object for preprocessing image data.

    Attributes:
    batch_size (int): The batch size for the data loaders.
    resize (int): The size to which the images should be resized.
    train_size (float): The proportion of the dataset to use for training.
    image_url_for_std (str): The URL of the images to use for calculating mean and standard deviation.
    image_url_for_train (str): The URL of the images to use for training.
    num_workers (int): The number of workers to use for loading the data.
    prediction_data (bool): A flag indicating whether the data is for prediction or not.
    """
    batch_size: int = 32
    resize:int = 224
    train_size: float = 0.8
    image_url_for_std: str = './plant-seedlings-classification/train/*/*.*'
    image_url_for_train: str = './plant-seedlings-classification/train/'
    num_workers:int = os.cpu_count()
    prediction_data:bool = False

class TrainingConfiguration:
    """
    A configuration object for training a PyTorch model.

    Attributes:
    model_name (str): The name of the model.
    epochs (int): The number of epochs to train the model.
    learning_rate (float): The learning rate for the optimizer.
    loss_criteron (nn.Module): The loss criterion to use for training.
    model (nn.Module): The PyTorch model to train.
    optimizer (type[torch.optim.Optimizer]): The optimizer to use for training.
    """
    model_name: str = 'resnet_100_epochs'
    epochs: int=150
    learning_rate: float = 0.001
    loss_criteron :nn = nn.CrossEntropyLoss()
    model: nn.Module = ResNet().to(device)
    optimizer: torch.optim = torch.optim.Adam
    
