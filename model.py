import torch 
from torch import nn
import torch.nn.functional as F
from config import width, height, numberOfResidualBlocks, numberOfFilters, dropoutRate, numberOfHiddenLayers, numberOfMapsPerPlayer, numberOfNeurons, valueHeadFilters, policyHeadFilters, performBatchNorm
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
        # Convolutional filters
        self.conv1 = nn.Conv2d(numberOfFilters, valueHeadFilters, 1)
        self.bn1 = nn.BatchNorm2d(valueHeadFilters)
        
        # Dropout layers
        self.dropoutLayers = nn.ModuleList([nn.Dropout(p = dropoutRate) for _ in range(numberOfHiddenLayers)])
        
        # Hidden layers
        hiddenLayers = [nn.Linear(valueHeadFilters * (width + 1) * (height + 1), numberOfNeurons)]
        for _ in range(numberOfHiddenLayers - 2):
            hiddenLayers.append(nn.Linear(numberOfNeurons, numberOfNeurons))

        hiddenLayers.append(nn.Linear(numberOfNeurons, 1))
    
        self.hiddenLayers = nn.ModuleList(hiddenLayers)
        
    def forward (self, x: Tensor) -> Tensor:
        """ Pass data through the value head of the network """
        # Pass through value filters
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropoutLayers[0](torch.flatten(x, start_dim = 1))
        
        # Pass through linear layers
        for do, h in zip(self.dropoutLayers[1:], self.hiddenLayers[:-1]):
            x = do(F.relu(h(x)))

        return torch.tanh(self.hiddenLayers[-1](x))
        
class PolicyHead (nn.Module):
    
    def __init__ (self):
        super(PolicyHead, self).__init__()
        """ Initializes the policy head of the network """
        # Convolutional filters
        self.conv1 = nn.Conv2d(numberOfFilters, policyHeadFilters, 1)
        self.bn1 = nn.BatchNorm2d(policyHeadFilters)
        
        # Dropout layers
        self.dropoutLayers = nn.ModuleList([nn.Dropout(p = dropoutRate) for _ in range(numberOfHiddenLayers)])
        
        # Hidden layers
        hiddenLayers = [nn.Linear(policyHeadFilters * (width + 1) * (height + 1), numberOfNeurons)]
        for _ in range(numberOfHiddenLayers - 2):
            hiddenLayers.append(nn.Linear(numberOfNeurons, numberOfNeurons))

        hiddenLayers.append(nn.Linear(numberOfNeurons, 7))
    
        self.hiddenLayers = nn.ModuleList(hiddenLayers)
        
    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        """ Pass data through the policy head of the network """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropoutLayers[0](torch.flatten(x, start_dim = 1))

        # Pass data through the hidden layers.
        for do, h in zip(self.dropoutLayers[1:], self.hiddenLayers[:-1]):
            x = do(F.relu(h(x)))

        if training:
            # If training is true return the logits
            return self.hiddenLayers[-1](x)

        else:
            # If training is false return the policy vector
            return torch.exp(torch.log_softmax(self.hiddenLayers[-1](x), dim = 1))
        
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
        self.valueHead = ValueHead()
        
    def forward (self, x: Tensor, training: bool = False) -> Tensor:
        """ Pass data through the network """
        # Pass the data through the residual part of the network
        x = F.relu(self.bn1(self.conv1(x)))
        for r in self.residualBlocks:
            x = r(x)
            
        # Pass the data through value head and policy head
        p = self.policyHead(x, training = training)
        
        v = self.valueHead(x)
        return (p, v) # [x, 7], [x, 1]
    
    def numberOfParameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

if (__name__ == '__main__'):
    m = Net()
    
    print(m.forward(torch.zeros((108, 3, 6, 7)))[1].shape)
    # model = Net().cuda()
    # from time import time 
    
    # t1 = time()
    # for i in range(50):
        # t = torch.randn(20, 2 * numberOfMapsPerPlayer + 1, 6, 7).to("cuda")
        # t = model(t)
    # print(time() - t1)