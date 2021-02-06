from typing import List
from model import Net
from state import *
from mcts import MCTS
from utils import loadLatetestModel
import numpy as np
from config import width
from tqdm import tqdm
from logger import info
from config import numberOfEvaluationGames, rooloutsDuringEvaluation
import random

def generateInitialStates (states: List[np.array]) -> List[np.array]:
    """ Generates a list of initial states """
    # Play the first move
    initialStates = []
    for i in range(width):
        for j in range(width):
            initialStates.append(makeMove(makeMove(states[i], i), j))

    return initialStates
    
def playGame (state: np.array, m1: MCTS, m2: MCTS) -> int:
    """ Returns 1 if m1 won the game, 0 if its a draw and -1 if m2 won """
    while (checkIfGameIsWon(state) == -1):
        if ((state[-1][0][0]) == 1): 
            # M1 plays a move
            state = makeMove(state, m1.getMove(state, rooloutsDuringEvaluation)) 
            
        else:
            # M2 plays a move
            state = makeMove(state, m2.getMove(state, rooloutsDuringEvaluation)) 
    
    m1.reset()
    m2.reset()
    
    # Check who won or drew the game.
    if (state[-1][0][0] == 0): 
        # M1 played the last move
        return checkIfGameIsWon(state)
    else:
        # M2 played the last move
        return -checkIfGameIsWon(state)

def evaluateModel (model: Net) -> int:
    """ Evaluates a model against the model which was saved last. """
    print("Evaluating model:")
    best = loadLatetestModel()
    
    # Initialize search trees 
    bestMCTS = MCTS(best)
    modelMCTS = MCTS(model)
    
    result = 0
    # Play games against the old model
    states = generateInitialStates([generateEmptyState() for _ in range(width)])
    for s in tqdm(random.sample(states, numberOfEvaluationGames)):
        
        # Actually play the games
        result += playGame(s, modelMCTS, bestMCTS)
        result -= playGame(s, bestMCTS, modelMCTS)
        
        # Reset each MCTS 
        modelMCTS.reset()
        bestMCTS.reset()
    
    # print(f" - Result from evaluation: {result}")
    info(f"Result from evaluation: {result}")
    return result