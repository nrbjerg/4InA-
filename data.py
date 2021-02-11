from logger import error
from mcts import MCTS
from model import Net, device
import torch
from state import flipGameState, makeMove, generateEmptyState, checkIfGameIsWon
from torch import from_numpy
import numpy as np
from numpy import random
from typing import Any, List
from config import numberOfGames, rooloutsDuringTraining, width, height, trainingOnGPU, tau
from tqdm import tqdm
import json 

def assignRewards(datapoints: List[List[np.array]], reward: float) -> List[List[np.array]]:
    """
        Args:
            - datapoints: A list of datapoints in the form [[state, policy, reward]] where reward is currently none
        Returns:
            - The list: [[state, policy, reward]] where reward is set to either (reward or -reward).
    """
    n = len(datapoints)
    
    for i in range(n):
        datapoints[n - i - 1][-1] = reward
        reward = -reward # The next state will be from the other players view (thus -reward)
    
    return datapoints

def addMirrorImages (datapoints: List[List[np.array]]) -> List[List[np.array]]:
    """ 
        Args: 
            - datapoints: A list of datapoints, in the form [[state, policy, reward]].
        Returns:
            - The list of datapoints with extra mirror images, specifically: 
            [[mirror(state), mirror(policy), reward]] + [[state, policy, reward]].
    """
    mirrors = []
    
    for d in datapoints:
        mirrors.append([flipGameState(d[0]), np.flip(d[1], axis = 1), d[2]])

    return datapoints + mirrors

def executeEpisode (mcts: MCTS) -> List[List[np.array]]:
    """ 
        Args: 
            - mcts: A montecarlo tree search algorithm
        Returns:
            - A list of datapoints, specifically: [[state, policy, reward]]
    """
    datapoints = []
    state = generateEmptyState()
    
    for move in range (width * height):      
        # Compute the probabilities & append the datapoint to the list.
        probs = mcts.getActionProbs(state, rooloutsDuringTraining)
        datapoints.append([state, probs, None]) 
        probs = probs.flatten()
        
        # Chose move (if the number of moves is < tau, play deterministically)
        if (move < tau):
            move = random.choice(len(probs), p = probs)
        else:
            move = np.argmax(probs)
        
        # Make the new move, check for rewards
        state = makeMove(state, move)
        reward = checkIfGameIsWon(state)
        
        if (reward != -1):
            # Assign rewards & add mirror images of the states
            return addMirrorImages(assignRewards(datapoints, reward))
        

def stackDataset (dataset: List[List[np.array]]) -> (np.array):
    """ Simpely stacks the dataset """
    arrays = []
    for idx in range(len(dataset[0])):
        arrays.append([item[idx] for item in dataset])
    
    return tuple([np.stack(array) for array in arrays])

def createDataset (model: Net, iteration: int) -> (np.array):
    """ 
        Args:
            - model: The neural network
            - iteration: The current iteration (used for enabled / disabeling value head in mcts)
        Returns:
            - A dataset in the form of a tuple of numpy arrays specifically (states, policies, rewards)
    """
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
    """ 
        Args: 
            - Dataset as a list of numpy arrays, specifically [states, policies, rewards]
        Returns:
            - The dataset converted to pytorch tensors moved to the gpu if needed, specifically (states, probabilities & rewards)
    """
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

def saveDataset (dataset: List[List[np.array]]):
    """ Saves the dataset to the datasets directory. """
    jsonObject = {}
    for idx, d in enumerate(dataset):
        jsonObject[str(idx)] = {"s": d[0].tolist(),
                                "p": d[1].tolist(),
                                "r": d[2].tolist()}
    
    # Save the data to the data.json file
    with open("data/data.json", "w") as jsonFile:
        json.dump(jsonObject, jsonFile)
        
def loadDataset () -> List[List[np.array]]:
    """ Loads the dataset from the datasets directory. """
    # Open json file and load it's contents
    try:
        with open("data/data.json", "r") as jsonFile:
            data = json.load(jsonFile)
            
            dataset = []
            for key in data.keys():
                dataset.append([np.array(data[key]["s"], dtype = "float32"), 
                                np.array(data[key]["p"], dtype = "float32"), 
                                np.array(data[key]["r"], dtype = "float32")])
                
            return dataset
    except FileNotFoundError:
        return []