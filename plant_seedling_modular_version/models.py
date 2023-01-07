from torch import nn 
from torchvision import models  
import warnings
warnings.filterwarnings("ignore")


class ResNet(nn.Module):
    """A PyTorch implementation of a ResNet model.
    
    This implementation uses a pre-trained ResNet-101 model and replaces the
    final fully-connected layer with a new linear layer with 12 output units.
    
    Args:
        nn (nn.Module): A PyTorch neural network module.
    
    Attributes:
        resnet (nn.Sequential): A PyTorch Sequential container for the 
            pre-trained ResNet-101 model, with the final two layers removed.
        Linear (nn.Linear): A PyTorch linear layer with 100352 input units
            and 12 output units.
    """
    def __init__(self):
        """Initializes the ResNet model.
        
        Initializes the resnet attribute as a PyTorch Sequential container 
        for the pre-trained ResNet-101 model, with the final two layers removed.
        Initializes the Linear attribute as a PyTorch linear layer with 
        100352 input units and 12 output units.
        """
        super().__init__()
        self.resnet = nn.Sequential(*(list(models.resnet101(pretrained=True).children())[:-2]))
        self.Linear = nn.Linear(in_features=100352, out_features=12)
    
    def forward(self, X):
        """Forward pass of the ResNet model.
        
        Args:
            X (torch.Tensor): A batch of input data with shape 
                (batch_size, num_channels, height, width).
        
        Returns:
            torch.Tensor: A batch of model output with shape 
                (batch_size, num_outputs).
        """
        X =  self.resnet(X)
        X = X.view(X.shape[0], -1 )
        X = self.Linear(X)
        return X
    
