from mcts import MCTS
from model import Net, device
import torch
from state import flipGameState, makeMove, generateEmptyState, checkIfGameIsWon
from torch import from_numpy
import numpy as np
from numpy import random
from typing import List
from config import numberOfGames, rooloutsDuringTraining, width, trainingOnGPU
from tqdm import tqdm

from time import time

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