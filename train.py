from numba.core.decorators import njit
from numpy.core.numeric import cross
from utils import loadLatetestModel, saveModel
from evaluate import evaluateModel
from typing import List
import torch
from torch import nn, Tensor
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam
from model import Net, device
from config import learningRate, disableValueHead, numberOfGames, batchSize, epochs, rooloutsDuringTraining, iterations, trainingOnGPU, width, mctsGPU, window
from state import generateEmptyState, makeMove, checkIfGameIsWon, flipGameState
from numpy import random
from mcts import MCTS
import numpy as np
from tqdm import tqdm, trange
from logger import info, error

# For debuging & testing:
from state import getStringRepresentation
from time import time 

mse = nn.MSELoss()

def crossEntropy (pred: Tensor, softTargets: Tensor) -> float:
    # TODO: There is something wrong with this loss function
    return torch.mean(torch.sum(-softTargets * F.log_softmax(pred, dim = 1), 1))
    
def criterion (output: (Tensor), target: (Tensor)) -> Tensor:
    """ Computes the loss using the output & target """
    valueLoss = mse(output[1], target[1])
    policyLoss = mse(output[0], target[0]) # CROSS ENTROPY: -(target[0] * torch.log(output[0])).sum(dim = 1).mean()
    return valueLoss + policyLoss

def assignRewards(datapoints: List[np.array], reward: float) -> List[np.array]:
    """ Assigns the reward to each data point """
    n = len(datapoints)
    
    for i in range(n):
        # print(getStringRepresentation(datapoints[i][0]), f"\nreward: {reward}", "\n")
        datapoints[n - i - 1][-1] = reward
        reward = -reward # The next state will be from the other players view (thus -reward)
    
    return datapoints

def addMirrorImages (datapoints: List[np.array]) -> List[np.array]: # TODO: Could be added to optimize training (amount of data needed vs learning)
    """ Adds the mirror images to the training data, thus increasing the number of datapoints pr. iteration """
    # NOTE: This could be destabilizing training
    mirrors = []
    
    for d in datapoints:
        mirrors.append([flipGameState(d[0]), np.flip(d[1], axis = 1), d[2]])

    return datapoints + mirrors

def executeEpisode (mcts: MCTS) -> List[np.array]:
    """ Plays a game with the neural network while storing the states as well as the actual policy vectors """
    datapoints = []
    state = generateEmptyState()
    
    while True:
        for _ in range(rooloutsDuringTraining):
            mcts.search(state)
        
        probs = mcts.getActionProbs(state, rooloutsDuringTraining)
        datapoints.append([state, probs, None]) 
        probs = probs.flatten()
        move = random.choice(len(probs), p = probs)
        state = makeMove(state, move)
        reward = checkIfGameIsWon(state)
        
        if (reward != -1):
            datapoints = addMirrorImages(assignRewards(datapoints, reward))
            return datapoints
        
        mcts.reset() # Remove old mcts states

def stackDataset (dataset: List[List[np.array]]) -> (np.array):
    """ Stacks the numpy arrays """
    arrays = []
    for idx in range(len(dataset[0])):
        arrays.append([item[idx] for item in dataset])
    
    return tuple([np.stack(array) for array in arrays])

def createDataset (model: Net, iteration: int) -> (np.array):
    """ Creates new training dataset """
    print("Creating dataset:")
    # Initialize montecarlo tree search
    mcts = MCTS(model, iteration = iteration)
    
    # Generate training data
    dataset = []
    for _ in tqdm(range(numberOfGames), unit = "game"):
        dataset += executeEpisode(mcts)
        mcts.reset() # Remove old states in mcts
        
    # Shuffle for better training
    info(f"Created dataset of {len(dataset)} datapoints.")
    return stackDataset(dataset)

def convertTrainingDataToTensors (dataset: List[np.array]) -> (torch.Tensor):
    """ Converts a dataset of states, policies and rewards into torch tensors used for training """
    tensors = [torch.from_numpy(array).float() for array in dataset]
    
    # Move arrays to gpu if needed
    if (trainingOnGPU == True):
        for i in range(3):
            tensors[i] = tensors[i].to(device)

    # Reshape some of the tensors
    n = len(tensors[0])
    states = tensors[0]
    probs = torch.reshape(tensors[1], (n, width))
    rewards = torch.reshape(tensors[2], (n, 1))
        
    return states, probs, rewards

def updateWeights (model: Net, states: torch.Tensor, probs: torch.Tensor, rewards: torch.Tensor) -> Net:
    """ 
        Updates the weights of the neural network based on the training data 
            x: states,
            ys: policies and rewards 
    """
    print("Updating weights:")
    # 1.1 Set model to training mode 
    model.train()
    
    # 1.2 Move model to gpu or back to cpu
    if (mctsGPU == False and trainingOnGPU == True): 
        model.cuda()
        
    elif (mctsGPU == True and trainingOnGPU == False):
        model.cpu()
    
    # 2. Train the neural network
    with trange(epochs, unit = "epoch") as t:
        for e in t:
            optimizer = Adam(model.parameters(), lr = learningRate)  
            
            permutation = torch.randperm(states.size()[0])
            
            totalValueLoss = 0
            totalPolicyLoss = 0
            for idx in range(0, states.size()[0], batchSize):

                # 2.1 Load minibatch traning data
                indices = permutation[idx:idx + batchSize]
                s = states[indices]
                p = probs[indices]
                r = rewards[indices]
                
                # 2.2 Pass data though network & update weights using the optimizer
                optimizer.zero_grad()
                predictions = model(s)
                
                # Check outputs for NaN
                if (torch.sum(predictions[0] != predictions[0])): error("nan in policy")
                elif (torch.sum(predictions[1] != predictions[1])): error("nan in value")
                
                policyLoss = mse(predictions[0], p) # TODO: Update this to another loss function
                valueLoss = 0
                if (disableValueHead == False):
                    valueLoss = mse(predictions[1], r)
                    loss = policyLoss + valueLoss
                else: 
                    loss = policyLoss
                
                loss.backward()
                optimizer.step()
                
                # 2.3 Update the total loss
                totalPolicyLoss += policyLoss.item()
                if (disableValueHead == False):
                    totalValueLoss += valueLoss.item()

            # totalLoss /= (states.size()[0] // batchSize) if (states.size()[0] >= batchSize) else 1 # Get the average loss pr mini batch
            if (disableValueHead == False):
                t.set_postfix(pl = totalPolicyLoss, vl = totalValueLoss)
            else: 
                t.set_postfix(pl = totalPolicyLoss)
                
            if (e == epochs  - 1):
                if (disableValueHead == False):
                    info(f"Ended with losses: \n - value loss: {totalValueLoss:.2f}\n - policy loss: {totalPolicyLoss:.2f}")
                else:
                    info(f"Ended with losses: \n - policy loss: {totalPolicyLoss:.2f}")
                    
    if (mctsGPU == False and trainingOnGPU == True):
        model.cpu()

    return model

def train(model: Net, fromScratch: bool):
    """ Trains the model """
    saveModel(model, "0.pt")
    datasets = []
    for iteration in range(iterations):
        print(f"\nAt iteration: {iteration + 1} / {iterations}")
        info(f"\nAt iteration: {iteration + 1} / {iterations}")
        
        # Create new dataset and append it to datasets
        states, probs, rewards = createDataset(model, iteration)
        datasets.append([states, probs, rewards])
        
        # Remove old datapoints
        while (len(datasets) > window(iteration)):
            datasets.pop(0)
        
        # Concatenate states, probs & rewards for training 
        data = [[] for _ in range(3)]
        for d in datasets:
            for idx, val in enumerate(d):
                data[idx].extend(val)
                
        data = [np.stack(d) for d in data]
        info(f"There is currently {data[0].shape[0]} datapoints in training data, from a maximum of {window(iteration)} games")
        states, probs, rewards = convertTrainingDataToTensors(data)
        
        model = updateWeights(model, states, probs, rewards)

        # Evaluate the model and get the percentage of games won by the new model
        percentage = evaluateModel(model) 
        if (percentage > 50.0):
            saveModel(model, f"{iteration + 1}.pt")
        else:
            datasets.pop() # Remove the last entry from the dataset 
            model = loadLatetestModel() # Load better model
            

if __name__ == '__main__':
    model = Net()
    train(model, False)