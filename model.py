import torch 
from torch import nn
import torch.nn.functional as F
from config import numberOfResidualBlocks, disableValueHead, numberOfFilters, dropoutRate, numberOfMapsPerPlayer, numberOfNeurons, valueHeadFilters, policyHeadFilters, performBatchNorm
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ResidualBlock (nn.Module):
    
    def __init__ (self):
        super(ResidualBlock, self).__init__()
        """ Initializes a residual block """
        # Initialize convolutional layers
        self.conv1 = nn.Conv2d(numberOfFilters, numberOfFilters, 2, padding = 1)
        self.conv2 = nn.Conv2d(numberOfFilters, numberOfFilters, 2)
        
        # Performs batch norm
        if (performBatchNorm == True):
            self.bn1 = nn.BatchNorm2d(numberOfFilters)
            self.bn2 = nn.BatchNorm2d(numberOfFilters)
        
    def forward (self, x: Tensor) -> Tensor:
        """ Pass the tensor x through the residual block """
        if (performBatchNorm == True):
            fx = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        else:
            fx = self.conv2(F.relu(self.conv1(x)))
        return F.relu(fx + x)

class ValueHead (nn.Module):
    
    def __init__ (self):
        """ Initializes the value head of the network """
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(numberOfFilters, valueHeadFilters, 1)
        self.bn1 = nn.BatchNorm2d(valueHeadFilters)
        self.do1 = nn.Dropout(p = dropoutRate)
        self.h1 = nn.Linear(valueHeadFilters * 7 * 8, numberOfNeurons)
        self.do2 = nn.Dropout(p = dropoutRate)
        self.h2 = nn.Linear(numberOfNeurons, 1)
    
    def forward (self, x: Tensor) -> Tensor:
        """ Pass data through the value head of the network """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.do1(torch.flatten(x, start_dim = 1))
        x = self.do2(F.relu(self.h1(x)))
        return torch.tanh(self.h2(x))
        
class PolicyHead (nn.Module):
    
    def __init__ (self):
        super(PolicyHead, self).__init__()
        """ Initializes the policy head of the network """
        self.conv1 = nn.Conv2d(numberOfFilters, policyHeadFilters, 1)
        self.bn1 = nn.BatchNorm2d(policyHeadFilters)
        self.do1 = nn.Dropout(p = dropoutRate)
        self.h1 = nn.Linear(policyHeadFilters * 7 * 8, numberOfNeurons)
        self.do2 = nn.Dropout(p = dropoutRate)
        self.h2 = nn.Linear(numberOfNeurons, 7)
        
    def forward (self, x: Tensor) -> Tensor:
        """ Pass data through the policy head of the network """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.do1(torch.flatten(x, start_dim = 1))
        x = self.do2(F.relu(self.h1(x)))
        return torch.exp(F.log_softmax(self.h2(x), dim = -1))
        
class Net (nn.Module):
    
    def __init__ (self):
        super(Net, self).__init__()
        """ Initializes the neural network """
        # Shared network
        self.conv1 = nn.Conv2d(2 * numberOfMapsPerPlayer + 1, numberOfFilters, 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(numberOfFilters)
        self.residualBlocks = nn.ModuleList([ResidualBlock() for i in range(numberOfResidualBlocks)])
        
        # Policy & value head
        self.policyHead = PolicyHead()
        
        if (disableValueHead == False):
            self.valueHead = ValueHead()
        
    def forward (self, x: Tensor) -> Tensor:
        """ Pass data through the network """
        # Pass the data through the residual part of the network
        x = F.relu(self.bn1(self.conv1(x)))
        for r in self.residualBlocks:
            x = r(x)
        # Pass the data through value head and policy head
        p = self.policyHead(x)
        if (disableValueHead == False):
            v = self.valueHead(x)
            return (p, v) # [x, 7], [x, 1]
        else:
            return (p, torch.zeros(p.shape[0], 1))
        
    
    def numberOfParameters (self) -> int:
        return sum([p.numel() for p in model.parameters()])
        
if (__name__ == '__main__'):
    model = Net()
    probs, rewards = model(torch.randn(10, 2 * numberOfMapsPerPlayer + 1, 6, 7))
    print(probs, rewards)
    print(model.numberOfParameters())