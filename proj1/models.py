# Import Pytorch Package 
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class BaselineNet(nn.Module):
    """
    A neural network without any convolution layer as the baseline.
    """
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(392, 160)
        self.fc2 = nn.Linear(80, 128)
        self.fc3 = nn.Linear(64, 2)
        
        self.bn1 = nn.BatchNorm1d(160)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        """
        General structure of 1 layer:
            Input -> Dropout -> Linear -> BatchNorm -> Activation(ReLu) -> Maxpooling -> Output
        """
        N = x.size()[0]
        # 1st layer
        x = F.relu(self.bn1(self.fc1(x.view(N, -1))))
        x = F.max_pool1d(x.view(N, 1, -1), kernel_size=2, stride=2)
        
        # 2nd layer
        x = F.dropout(x, p = 0.5)
        x = F.relu(self.bn2(self.fc2(x.view(N, -1))))
        x = F.max_pool1d(x.view(N, 1, -1), kernel_size=2)
        
        # 3rd layer
        x = F.dropout(x, p = 0.5)
        x = F.relu(self.fc3(x.view(N, -1)))
        return x
    
class CNN(nn.Module):
    """
    Models with weight sharing.
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding = 3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm1d(720)
        self.fc1 = nn.Linear(720, 100)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x):
        """
        General structure of one layer:
            Input -> Convolution -> BatchNorm -> Activation(ReLu) -> Maxpooling -> Output
        """
        # 1st layer 
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)
        # 2nd layer 
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        # 3rd layer
        x = F.relu(self.fc1(self.bn3(x.view(x.size()[0], -1))))
        # 4th layer
        x = self.fc2(F.dropout(x)) 
        
        return x
    
class CNNAuxiliaryLoss(nn.Module):
    def __init__(self):
        """
        CNN Model with Auxiliary Loss
        """
        super(CNNAuxiliaryLoss, self).__init__()
        
        nb_hidden = 120
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding = 3)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=5, padding = 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(500, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        """
        General structure of one layer:
            Input -> Convolution -> BatchNorm -> Activation(ReLu) -> Maxpooling -> Output
        """
        # 1st layer
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)
        # 2nd layer
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        # 3rd layer
        x = F.relu(self.fc1(x.view(x.size()[0], -1)))
        
        # This layer is used for digit classification and introduce auxiliary loss
        xClassification = self.fc2(x)
        
        # Layer for generating true label
        x = self.fc3(F.relu(xClassification))
        x = x.view(int(xClassification.size()[0]/2), -1)
        
        return xClassification, x
    