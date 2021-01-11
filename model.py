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
    
    def __init__ (self, hidden_nodes: int = 64):
        """ Initializes a model for tic tac toe """
        super(Net, self).__init__()
        # Convolutional Layers 
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.c1 = nn.Conv2d(1, 8, 3)
        self.c2 = nn.Conv2d(8, 8, 2) # 1, 8, 4, 3                   ]
        
        self.hiddenLayer = nn.Linear(8 * 4 * 3, hidden_nodes)
        
        # Value and policy head
        self.policy = nn.Linear(hidden_nodes, 7)
        self.value = nn.Linear(hidden_nodes, 1)
        
    def forward (self, x: Tensor) -> (Tensor):
        """ Pass data through the network and return the expected policy and value """
        # Convolutional layers
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        
        # Flatten + Hidden layer
        x = x.view(-1, 8 * 4 * 3)
        x = F.relu(self.hiddenLayer(x))
        # Policy
        p = torch.sigmoid(self.policy(x))
        
        # Value 
        v = torch.tanh(self.value(x))  
           
        return p, v

    
if __name__ == "__main__":
    loss = nn.CrossEntropyLoss()
    inputs = torch.randn(3, 1, 6, 7, requires_grad=True).to(device)
    target = torch.empty(3, dtype=torch.long).random_(7).to(device)
    reward = torch.randn(3, 1).to(device)
    model = Net()
    model.cuda()
    # 2. Initialize optimizer
    optimizer = SGD(model.parameters(), lr = 0.001)
    
    # 3. Run training loop 
    epochs = 20_000
    printTime = epochs // 10
    MSEloss = nn.MSELoss()
    for e in range(epochs):
        optimizer.zero_grad()
        
        o1, o2 = model(inputs)
        l1 = loss(o1, target) # TODO: I think that the current loss function is fucked.
        l2 = MSEloss(o2, reward)
        l = l1 + l2
        l.backward()
        optimizer.step()
        
        if ((e + 1) % printTime == 0):
            print(f"  - Update! currently at epoch {e + 1} / {epochs}, with loss: {l.item():.6f}")
    
    # 4. Move model to cpu
    model.cpu()