from mcts import MCTS
from parallel import ParallelMCTS
from model import Net, device
import torch
from state import flipGameState, getStringRepresentation, makeMove, generateEmptyState, checkIfGameIsWon
import numpy as np
from numpy import random
from typing import List, Tuple
from config import numberOfGames, rooloutsDuringTraining, width, height, trainingOnGPU, tau, customReward, rewardDropOf
from tqdm import tqdm
import json 
from logger import logInfo 

def assignRewards(datapoints: List[List[np.array]], reward: float) -> List[List[np.array]]:
    """
        Args:
            - datapoints: A list of datapoints in the form [[state, policy, reward]] where reward is currently none
        Returns:
            - The list: [[state, policy, reward]] where reward is set to either (reward or -reward).
    """
    if (reward != 0.0):
        n = len(datapoints)

        for i in range(min(rewardDropOf, n)):
            datapoints[-(i + 1)][-1] = reward
            # The next state will be from the other players view (thus -reward)
            # Also the numerical value of the rewards should drop of after each move if using custom reward function
            reward = -np.sign(reward) / (i + 1) if customReward == True else -reward

    return datapoints

def addMirrorImages(datapoints: List[List[np.array]]) -> List[List[np.array]]:
    """ 
        Args: 
            - datapoints: A list of datapoints, in the form [[state, policy, reward]].
        Returns:
            - The list of datapoints with extra mirror images, specifically: 
            [[mirror(state), mirror(policy), reward]] + [[state, policy, reward]].
    """
    mirrors = [
        [flipGameState(d[0]), np.flip(d[1], axis=1), d[2]] for d in datapoints
    ]

    return datapoints + mirrors
    
def stackDataset(dataset: List[List[np.array]]) -> np.array:
    """ Simpely stacks the dataset """
    arrays = [[item[idx] for item in dataset] for idx in range(3)] # Create a list with nested lists of gamestates, probs, rewards
    return tuple(np.stack(array) for array in arrays) # Stack the lists together

def choseMove (probs: np.array, numberOfMoves: int) -> int:
    """ Choses a move determined by the probs & the number of moves """
    return (
            random.choice(len(probs), p=probs)
            if (numberOfMoves < tau)
            else np.argmax(probs)
    )

def executeEpisode(mcts: MCTS) -> List[List[np.array]]:
    """ 
        Args: 
            - mcts: A montecarlo tree search algorithm
        Returns:
            - A list of datapoints, specifically: [[state, policy, reward]]
    """
    datapoints = []
    state = generateEmptyState()

    for numberOfMoves in range (width * height):  
        # Compute the probabilities & append the datapoint to the list.
        probs = mcts.getActionProbs(state, rooloutsDuringTraining)
        datapoints.append([state, probs, 0.0]) # NOTE: The reward changed if reward != 0.0 (in the assign reward function)
        probs = probs.flatten()

        # Chose move (if the number of moves is < tau, play deterministically)
        move = choseMove(probs, numberOfMoves)

        # Make the new move, check for rewards
        state = makeMove(state, move)
        reward = checkIfGameIsWon(state)

        if (reward != -1):
            # Assign rewards & add mirror images of the states
            return assignRewards(datapoints, float(reward))
            # return addMirrorImages(assignRewards(datapoints, float(reward)))

def createDatasetWithNormalMCTS (model: Net, iteration: int) -> Tuple[np.array]:
    """ 
        Args:
            - iteration: The current iteration (used for enabled / disabeling value head in mcts)
        Returns:
            - A dataset in the form of a tuple of numpy arrays specifically (states, policies, rewards)
    """
    mcts = MCTS(model = model, iteration = iteration)

    # Generate training data
    dataset = []
    for _ in tqdm(range(numberOfGames), unit = " game"):
        dataset += executeEpisode(mcts)
        mcts.reset() # Remove old states in mcts

    return stackDataset(dataset)

def createDatasetWithParrallelMCTS (iteration: int) -> Tuple[np.array]:
    """ 
        Args:
            - iteration: The current iteration (used for enabled / disabeling value head in mcts)
        Returns:
            - A dataset in the form of a tuple of numpy arrays specifically (states, policies, rewards)
    """
    # FIXME: There is some major problems with this code
    print("Creating dataset:")
    # predictor.updateModel()
    # Initialize montecarlo tree search
    mcts = ParallelMCTS(iteration = iteration)

    # Generate training data
    states = [generateEmptyState() for _ in range(numberOfGames)]
    datapoints = [[] for _ in range(numberOfGames)]
    rewards = [-1 for _ in range(numberOfGames)]
    numberOfMoves = 0

    while -1 in rewards:     
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
                # For debuging purposes
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

    print(f"Created a dataset of {numberOfGames} games ({len(datapoints)} individual datapoints)")
    return stackDataset(sum(datapoints, []))

def convertTrainingDataToTensors (dataset: List[np.array], checkForNan: bool = False) -> (torch.Tensor):
    """ 
        Args: 
            - Dataset as a list of numpy arrays, specifically [states, policies, rewards]
        Returns:
            - The dataset converted to pytorch tensors moved to the gpu if needed, specifically (states, probabilities & rewards)
    """
    # Check wether or not the dataset contains nan
    if checkForNan: 
        logInfo(str([np.isnan(np.sum(d)) for d in dataset[:3]]))

    dataset = [array.astype("float32") for array in dataset]
    tensors = [torch.from_numpy(array) for array in dataset]
    
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

def saveDataset(dataset: List[List[np.array]]):
    """ Saves the dataset to the datasets directory. """
    jsonObject = {
        str(idx): {"s": d[0].tolist(), "p": d[1].tolist(), "r": d[2].tolist()}
        for idx, d in enumerate(dataset)
    }

    # Save the data to the data.json file
    with open("data/data.json", "w+") as jsonFile:
        json.dump(jsonObject, jsonFile)
        
def loadDataset() -> List[List[np.array]]:
    """ Loads the dataset from the datasets directory. """
    # Open json file and load it's contents
    try:
        with open("data/data.json", "r") as jsonFile:
            data = json.load(jsonFile)

            return [
                       [np.array(data[key]["s"], dtype="float32"),
                        np.array(data[key]["p"], dtype="float32"),
                        np.array(data[key]["r"], dtype="float32")]
                    for key in data.keys()]

    except FileNotFoundError:
        return []
    