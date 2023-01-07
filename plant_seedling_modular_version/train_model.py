"""
The model is trained here using all the pipelines and the final model is saved and performance is calculated.
"""
from training_steps import calculate_class_weights, train_model, val
from torch import nn 
import tqdm 
import torch 
from configuration import TrainingConfiguration, device
from preprocessing_data import preprocess_image_folder_data
import pandas as pd 
import warnings 
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



def full_training(train_config: TrainingConfiguration = TrainingConfiguration()):
    """
    Train and evaluate a PyTorch model on a given dataset.

    The model is trained for a specified number of epochs, and its performance is
    evaluated on both the training and test datasets. The model with the best
    performance on the test dataset is saved.

    Parameters:
    train_config (TrainingConfiguration, optional): A configuration object
        containing the model, optimizer, learning rate, and number of epochs.
        Default is an instance of TrainingConfiguration.

    Returns:
    tuple: A tuple containing the mean and standard deviation of the training data,
        the best model, and a DataFrame of the model's performance during training.
    """

    model = train_config.model
    optimizer = train_config.optimizer(params=model.parameters(), lr=train_config.learning_rate)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    f1_train_lis = []
    f1_test_lis = []
    current_best = 0
    mean, std, train, test, classes = preprocess_image_folder_data()
    
    print('Step 2: Calculating class weights')
    weights = calculate_class_weights(train)
    loss_criteron = nn.CrossEntropyLoss(weight=weights,reduction='mean').to(device)
    print('Step 3: Training Model')
    for epoch in tqdm.tqdm(range(train_config.epochs)):
        accuracy_train, loss_train, f1_train = train_model(model, train , loss_criteron, optimizer, classes)
        accuracy_test, loss_test, f1_test = val(model, test, loss_criteron, classes)
        if current_best < accuracy_test:
            current_best = accuracy_test
            torch.save(model.state_dict(), './saved_models/' + train_config.model_name +'_best_model.pth')
        train_loss.append(loss_train)
        train_acc.append(accuracy_train)
        f1_train_lis.append(f1_train)
        test_loss.append(loss_test)
        test_acc.append(accuracy_test)
        f1_test_lis.append(f1_test)
        print('Epoch:', epoch + 1, '/', train_config.epochs, '| train_acc:', round(accuracy_train,2),  '| f1_train:', round(f1_train,2) , 
            '| train_loss:', round(loss_train,2), ' | test_acc:', round(accuracy_test,2), '| f1_test:', round(f1_test,2),'| test_loss:', round(loss_test,2) )
    model_performance_dict = {
        'Train_Accuracy': train_acc,
        'Train_Loss': train_loss,
        'F1_Train': f1_train_lis,
        'Test_Accuracy': test_acc,
        'Test_Loss': test_loss,
        'F1_Test': f1_test_lis
    }
    performance = pd.DataFrame(model_performance_dict)
    model = torch.load('./saved_models/' + train_config.model_name +'_best_model.pth')
    return mean, std, model, performance

if __name__ == "__main__":
    full_training()

        
