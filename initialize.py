from typing import List, Tuple
from config import numberOfMapsPerPlayer
from state import flipPlayer, generateEmptyState, getStringRepresentation
import numpy as np
from model import Net, device
import pandas as pd 

df = pd.read_csv("games.csv")

array = df.to_numpy()

winner = array[:, -1]
positions = array[:, :-1]

def convertPositions(positions: np.array) -> Tuple[np.array]:
    """
        Args:
            - Positions: np.array a numpy array of positions
        Returns:
            - A Tuple of states and results
    """
    states, winners = [], []
    
    for row in positions:
        matrix = row.reshape((6, 7))
        winners.append(np.sum(matrix))
        state = generateEmptyState()
        
        # Convert to the actual state object
        for i in range(state.shape[1]):
            for j in range(state.shape[2]):
                if (matrix[i][j] == 1):
                    state[0][i][j] = 1.0
                    
                elif (matrix[i][j] == -1):
                    state[1][i][j] = 1.0

                if (matrix[i][j] != 0):
                    flipPlayer(state)
        
        states.append(state)
    
    return (np.stack(states), np.array(winners))
    
states, rewards = convertPositions(positions)
print(states.shape, rewards.shape)

import torch 
from torch.optim import Adam
from tqdm import trange
from torch.nn import MSELoss

states = torch.from_numpy(states).float()
rewards = torch.from_numpy(rewards).reshape(rewards.shape[0], 1).float()

model = Net()

model.train()
model.cuda()
mse = MSELoss()
bsize = 1024
# Train the neural network
with trange(10, unit = "epoch") as t:
    for e in t:
        optimizer = Adam(model.parameters(), lr = 0.001)  
        
        permutation = torch.randperm(states.size()[0])
        
        valueLoss = 0
        batches = 0
        for idx in range(0, states.size()[0], bsize):
            batches += 1
            # Load minibatch traning data
            indices = permutation[idx:idx + bsize]
            s = states[indices].to(device)
            r = rewards[indices].to(device)
            predictions = model(s)
            
            loss = mse(predictions[1], r)
            
            loss.backward()
            optimizer.step()
            
            # Update the total loss
            valueLoss += loss.item()
            
        valueLoss /= batches
        
        # Update progres bar
        t.set_postfix(vl = valueLoss)

from utils import saveModel
saveModel(model, "v.pt")