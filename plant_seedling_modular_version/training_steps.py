import torch 
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import F1Score
import numpy as np 
from configuration import device
from torch.utils.data import DataLoader


def calculate_class_weights(train_loader: DataLoader):
    """
    Calculate class weights for a PyTorch dataloader.

    The class weights are calculated using the 'balanced' option from
    sklearn.utils.class_weight.compute_class_weight.

    Parameters:
    train_loader (DataLoader): A PyTorch dataloader containing the training data.

    Returns:
    torch.tensor: A tensor of class weights, with one weight for each class.
    """
    targets = torch.tensor([])
    for batch, (X, y) in enumerate(train_loader):
        targets = torch.cat((targets, y), 0)

    class_weights=compute_class_weight(class_weight='balanced' , classes = np.unique(targets), y = targets.numpy())
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    return class_weights

def train_model(model: torch.nn.Module, train_loader: DataLoader, loss_criteron, optimizer: torch.optim, classes: list):
    """
    Train a PyTorch model on a given dataset.

    Parameters:
    model (torch.nn.Module): The PyTorch model to train.
    train_loader (DataLoader): A PyTorch dataloader containing the training data.
    loss_criterion: The criterion used to compute the loss.
    optimizer (torch.optim): The optimizer used to update the model's weights.
    classes (list): A list of class labels.

    Returns:
    tuple: A tuple containing the training accuracy, average loss, and F1 score.
    """
    model.train()
    loss_sum = 0
    total_correct = 0 
    f1 = F1Score(task="multiclass", num_classes=len(classes), average='micro' ).to(device)
    pred =  torch.tensor([]).to(device)
    target =  torch.tensor([]).to(device)

    for batch, (X, y) in enumerate(train_loader):
        y_logits =  model(X.to(device))
        loss = loss_criteron(y_logits, y.to(device))
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), 1)
        pred = torch.cat((pred, y_pred),0)
        target = torch.cat((target, y.to(device)), 0)
        loss_sum += loss.to('cpu').item()
        total_correct += torch.sum(torch.eq(y_pred, y.to(device))).to('cpu').item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    f1_score = f1(pred, target)
    accuracy = total_correct/len(train_loader.dataset)
    avg_loss = loss_sum/len(train_loader.dataset)
    return accuracy, avg_loss, f1_score.item()

def val(model: torch.nn.Module, test_loader: DataLoader, loss_criteron, classes: list):
    """
    Evaluate a PyTorch model on a given dataset.

    Parameters:
    model (torch.nn.Module): The PyTorch model to evaluate.
    test_loader (DataLoader): A PyTorch dataloader containing the test data.
    loss_criterion: The criterion used to compute the loss.
    classes (list): A list of class labels.

    Returns:
    tuple: A tuple containing the test accuracy, average loss, and F1 score.
    """
    model.eval()
    pred =  torch.tensor([]).to(device)
    target =  torch.tensor([]).to(device)
    loss_sum = 0
    total_correct = 0 
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            y_logits =  model(X.to(device))
            loss = loss_criteron(y_logits, y.to(device))
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            pred = torch.cat((pred, y_pred),0)
            target = torch.cat((target, y.to(device)), 0)
            loss_sum += loss.to('cpu').item()
            total_correct += torch.sum(torch.eq(y_pred, y.to(device))).to('cpu').item()
    f1 = F1Score(task="multiclass", num_classes=len(classes), average='micro' ).to(device)
    f1_score = f1(pred, target)    
    accuracy = total_correct/len(test_loader.dataset)
    avg_loss = loss_sum/len(test_loader.dataset)
    return accuracy, avg_loss, f1_score.item() 
