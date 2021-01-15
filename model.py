import os
import torch
from torch import nn
from torch import Tensor
from torch.optim import SGD, Adam
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def loss (output: (Tensor), target: (Tensor)):  
    crossEntropy = torch.sum(target[0] * torch.log(output[0]))
    mse = nn.MSELoss()(output[1], target[1])
    return mse - crossEntropy

class Net(nn.Module):
    
    def __init__ (self, numberOfHiddenNodes: int = 256):
        """ Initializes a model for tic tac toe """
        super(Net, self).__init__()
        # Convolutional Layers 
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.c1 = nn.Conv2d(1, 8, 3)
        self.c2 = nn.Conv2d(8, 8, 2) # 1, 8, 4, 3                   ]
        
        # Hidden layers
        self.h1 = nn.Linear(8 * 4 * 3, numberOfHiddenNodes)
        self.h2 = nn.Linear(numberOfHiddenNodes, numberOfHiddenNodes)
        
        # Value and policy head
        self.policy = nn.Linear(numberOfHiddenNodes, 7)
        self.value = nn.Linear(numberOfHiddenNodes, 1)
        
    def forward (self, x: Tensor) -> (Tensor):
        """ Pass data through the network and return the expected policy and value """
        # Convolutional layers
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        
        # Flatten + Hidden layers
        x = x.view(-1, 8 * 4 * 3)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        
        # Policy
        p = torch.sigmoid(self.policy(x))
        
        # Value 
        v = torch.tanh(self.value(x))  
           
        return p, v
    
    @property
    def numberOfTrainableParameters(self):
        """ Computes the number of trainable parameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def saveModel (model: Net, filename: str):
    """ Saves the current model """
    if (not os.path.exists("models")):
        os.mkdir("models")
        
    filepath = os.path.join(os.getcwd(), "models", filename)
    torch.save(model, filepath)

def loadModel(filename: str) -> Net:
    """ Loads a model from the models directory """
    return torch.load(os.path.join(os.getcwd(), "models", filename))

def loadLatetestModel () -> Net:
    """ Loads the latest model from the models directory """
    files = os.listdir(os.path.join(os.getcwd(), "models"))
    file = str(sorted([int(f.split(".")[0]) for f in files])[-1]) + ".pt"
    model = loadModel(file)
    return model