from typing import List, Tuple
from model import Net
from state import *
from mcts import MCTS
from utils import loadLatetestModel, loadModel
import numpy as np
from config import width
from tqdm import tqdm
from logger import info
from config import numberOfEvaluationGames, rooloutsDuringEvaluation
import random

def generateInitialStates () -> List[np.array]:
    """ 
        Returns:
            - A list of all possible states after the initial two moves.
    """
    initialStates = []
    for i in range(width):
        for j in range(width):
            initialStates.append(makeMove(makeMove(generateEmptyState(), i), j))

    return initialStates
    
def playGame (state: np.array, m1: MCTS, m2: MCTS, w: int, l: int) -> Tuple[int]:
    """ 
        Args:
            - State: The state from which the game will take place
            - M1 & M2: the two MCTS instances
            - w & l: number of wins & losses of the m1 player
            
        Returns:
            - A tuple of (w & l) incremented modified depending on which model won the game 
    """
    while (checkIfGameIsWon(state) == -1):
        if ((state[-1][0][0]) == 1): 
            # M1 plays a move
            state = makeMove(state, m1.getMove(state, rooloutsDuringEvaluation)) 
            
        else:
            # M2 plays a move
            state = makeMove(state, m2.getMove(state, rooloutsDuringEvaluation)) 
    
    m1.reset()
    m2.reset()
    
    reward = checkIfGameIsWon(state)
    # Check who won or drew the game.
    if (state[-1][0][0] == 0): 
        # M1 played the last move
        return (w + reward, l)
    else:
        # M2 played the last move
        return (w, l + reward)

def evaluateModel (model: Net, iteration: int, fast: bool = False, opponent: Net = None) -> float:
    """ 
        Creates two MCTS trees, one for the model passed as an argument, and one for
        the model loaded by the loadLatestModel function. It then pits them against each other.
        Returns:
            - The models winrate against the model loaded by the loadLatestModel function.
    """
    print("Evaluating model:")
    best = loadLatetestModel()[0]
    
    # Initialize search trees 
    if (opponent == None):
        bestMCTS = MCTS(best, iteration = iteration)
    else:
        bestMCTS = MCTS(opponent, iteration = iteration)
        
    modelMCTS = MCTS(model, iteration = iteration)
    
    # Play games against the old model
    wins, losses = 0, 0
    if (fast == False):
        states = generateInitialStates()
    else:
        states = [makeMove(generateEmptyState(), idx) for idx in range(7)]
        
    for s in tqdm(random.sample(states, (numberOfEvaluationGames if (fast == False) else 7))):
        # Actually play the games
        wins, losses = playGame(s.copy(), modelMCTS, bestMCTS, wins, losses)
        losses, wins = playGame(s.copy(), bestMCTS, modelMCTS, losses, wins)

    # Compute the winrate
    winrate = round((wins / (2 * (numberOfEvaluationGames if (fast == False) else 7))) * 100, 1)
          
    info(f"Winrate during evaluation : {winrate:.1f} %")
    return winrate
      
if (__name__ == "__main__"):
    print(evaluateModel(loadModel("0.pt"), 0, fast = True))
    print(evaluateModel(loadModel("0.pt"), 200, fast = True))