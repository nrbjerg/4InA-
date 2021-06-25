from mcts import MCTS
from parallel import ParallelMCTS
from model import Net, device
import torch
from state import flipGameState, getStringRepresentation, makeMove, generateEmptyState, checkIfGameIsWon
import numpy as np
from numpy import random
from typing import List
from config import numberOfGames, rooloutsDuringTraining, width, height, trainingOnGPU, tau, customReward, rewardDropOf
from tqdm import tqdm
import json 
import math
from utils import loadLatetestModel, saveModel
from predictor import predictor

def sigmoid (x: float) -> float:
    # Just the standart sigmoid function
    return 1 / (1 + math.exp(x))

def assignRewards(datapoints: List[List[np.array]], reward: float) -> List[List[np.array]]:
    """
        Args:
            - datapoints: A list of datapoints in the form [[state, policy, reward]] where reward is currently none
        Returns:
            - The list: [[state, policy, reward]] where reward is set to either (reward or -reward).
    """
    n = len(datapoints)
    
    if (reward != 0.0):
        for i in range(n):
            if (i >= rewardDropOf):
                datapoints[n - i - 1][-1] = reward
                # The next state will be from the other players view (thus -reward)
                # Also the numerical value of the rewards should drop of after each move
                if customReward == True: 
                    reward = 1 - (sigmoid(-i // 2) / 2 - 0.2) if (i % 2) == 0 else -reward
                else:
                    reward = -1.0
            else: 
                datapoints[n - i - 1][-1] = 0

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
    
def stackDataset (dataset: List[List[np.array]]) -> (np.array):
    """ Simpely stacks the dataset """
    arrays = []
    for idx in range(len(dataset[0])):
        arrays.append([item[idx] for item in dataset])
    
    return tuple([np.stack(array) for array in arrays])

def createDataset (iteration: int) -> (np.array):
    """ 
        Args:
            - model: The neural network
            - iteration: The current iteration (used for enabled / disabeling value head in mcts)
        Returns:
            - A dataset in the form of a tuple of numpy arrays specifically (states, policies, rewards)
    """
    print("Creating dataset:")
    predictor.updateModel()
    # Initialize montecarlo tree search
    mcts = ParallelMCTS(iteration = iteration)
    
    # Generate training data
    states = [generateEmptyState() for _ in range(numberOfGames)]
    datapoints = [[] for _ in range(numberOfGames)]
    rewards = [-1 for _ in range(numberOfGames)]
    numberOfMoves = 0
    
    while (any([True if r == -1 else False for r in rewards])):     
        # Compute the probabilities & append the datapoint to the list.
        s, indexes, n = [], [], 0
        for idx, state in enumerate(states):
            if (rewards[idx] == -1):
                s.append(state)
                indexes.append(idx)
                n += 1
                
        probs = mcts.getActionProbs(s, rooloutsDuringTraining)
        
        for idx, (state, p) in enumerate(zip(s, probs)):
            datapoints[idx].append([state, p, None]) 
            
        moves = []
        for idx, p in enumerate(probs):
            p = p.flatten()
            if (np.isnan(np.sum(p))):
                print(getStringRepresentation(s[idx]))
                
            # Chose move (if the number of moves is < tau, play deterministically)
            if (numberOfMoves < tau):
                moves.append(random.choice(len(p), p = p))
            else:
                moves.append(np.argmax(p))
        
        for idx, move in enumerate(moves):
            # Make the new move, check for rewards
            states[indexes[idx]] = makeMove(states[indexes[idx]], move)
            rewards[indexes[idx]] = checkIfGameIsWon(states[indexes[idx]])
    
    # Assign rewards and add mirror images
    for idx in range(numberOfGames):
        datapoints[idx] = addMirrorImages(assignRewards(datapoints[idx], rewards[idx]))
      
    return stackDataset(sum(datapoints, []))

def convertTrainingDataToTensors (dataset: List[np.array]) -> (torch.Tensor):
    """ 
        Args: 
            - Dataset as a list of numpy arrays, specifically [states, policies, rewards]
        Returns:
            - The dataset converted to pytorch tensors moved to the gpu if needed, specifically (states, probabilities & rewards)
    """
    print([np.isnan(np.sum(d)) for d in dataset[:3]])
    dataset = [array.astype("float32") for array in dataset]
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
    
if (__name__ == "__main__"):

    # model, iteration = loadLatetestModel()
    # if (iteration == 0):
        # saveModel(model, "0.pt")
    
    # Code for profiling: 
    import sys 
    # import cProfile
    
    sys.stdout = open("profile.txt", "w")

    createDataset(0)
    
    # sys.stdout.close()