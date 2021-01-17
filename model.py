import os
import torch
from torch import nn
from torch import Tensor
from torch.optim import SGD, Adam
import torch.nn.functional as F
import numpy as np
from config import numberOfChannels, numberOfResidualBlocks, dropoutRate

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def loss (output: (Tensor), target: (Tensor)):  
    crossEntropy = torch.sum(target[0] * torch.log(output[0]))
    mse = nn.MSELoss()(output[1], target[1])
    return mse - crossEntropy

class ResidualBlock(nn.Module):
    
    def __init__ (self, channels: int, kernelSize: int, p: int):
        """ Initializes a residual block """
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(channels, channels, kernelSize, padding = p)
        self.c2 = nn.Conv2d(channels, channels, kernelSize, padding = p)
        

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """ Sends the tensor through the layer """
        residual = x
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x) + residual)
        return x
    
class Net(nn.Module):
    
    def __init__ (self):
        """ Initializes a model for tic tac toe """
        super(Net, self).__init__()
        # Convolutional Layers 
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.firstConvolution = nn.Conv2d(1, numberOfChannels, 3, padding = 1)
        
        self.residualBlocks = nn.ModuleList([ResidualBlock(numberOfChannels, 3, 1) for i in range(numberOfResidualBlocks)])
        
        self.outputFromResidualLayers = numberOfChannels * 6 * 7 # Output dimension from the residual blocks
        
        # self.c2 = nn.Conv2d(8, 8, 2) # 1, 8, 4, 3
        
        # Hidden layers
        # self.h1 = nn.Linear(8 * 4 * 3, numberOfHiddenNodes)
        # self.h2 = nn.Linear(numberOfHiddenNodes, numberOfHiddenNodes)
        
        # Value and policy head
        self.policyHiddenLayer1 = nn.Linear(self.outputFromResidualLayers, self.outputFromResidualLayers)
        self.policyDropoutLayer = nn.Dropout(dropoutRate)
        self.policyHiddenLayer2 = nn.Linear(self.outputFromResidualLayers, 7)
        
        self.valueHiddenLayer1 = nn.Linear(self.outputFromResidualLayers, self.outputFromResidualLayers)
        self.valueDropoutLayer = nn.Dropout(dropoutRate)
        self.valueHiddenLayer2 = nn.Linear(self.outputFromResidualLayers, 1)
        
    def forward (self, x: Tensor) -> (Tensor):
        """ Pass data through the network and return the expected policy and value """
        # Convolutional layers
        x = F.relu(self.firstConvolution(x))
        
        for r in self.residualBlocks:
            x = r(x)
        
        # Flatten + Hidden layers
        x = x.view(-1, self.outputFromResidualLayers) # Flatten the output of the residual blocks
        
        # Policy
        p = F.relu(self.policyHiddenLayer1(x))
        p = self.policyDropoutLayer(p)
        p = torch.sigmoid(self.policyHiddenLayer2(p))
        
        # Value 
        v = F.relu(self.valueHiddenLayer1(x))
        v = self.valueDropoutLayer(v)
        v = torch.tanh(self.valueHiddenLayer2(v))  
           
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

if __name__ == "__main__":
    model = Net()
    print(model(torch.randn(1, 1, 6, 7)))
    print(model.numberOfTrainableParameters)