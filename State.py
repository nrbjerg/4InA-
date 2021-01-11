import numpy as np
import copy
from numba import njit

@njit()
def checkIfGameIsWon (state: np.array) -> int:
    """ 
        Returns:
            1  if player (1) has won the game.
            0  if the game was a draw.
            -1 if the game hasn't ended
    """
    m, n = state.shape # Get the dimensions of the matrix 

    # 1. Check for horizontal alignments
    for i in range(m):
        for j in range(n - 3):
            if (np.sum(state[i, j : j + 4]) == 4.0): return 1 
            
    # 2. Check for vertical alignments
    for i in range(m - 3):
        for j in range(n):
            if (np.sum(state[i: i + 4, j]) == 4.0): return 1
     
    # 3. Check for diagonal alignments
    for i in range(m - 3):
        for j in range(n - 3):
            if (state[i, j] + state[i + 1, j + 1] + state[i + 2, j + 2] + state[i + 3, j + 3] == 4.0): return 1
            if (state[i, j + 3] + state[i + 1, j + 2] + state[i + 2, j + 1] + state[i + 3, j] == 4.0): return 1
    
    # 4. Check for draws (no available moves)
    if (np.sum(np.abs(state[0, :])) == n): return 0 
    
    # 5. If none of the above the game hasn't ended yet
    return -1

@njit()
def makeMove (state: np.array, idx: int) -> np.array:
    """ Copies the state and makes a move on the copy """
    state = state.copy()
    height = state.shape[0]
    for i in range(1, height + 1): # Play moves from the button up
        if (state[height - i][idx] == 0.0):
            # Play the move
            state[height - i][idx] = 1.0
            break # Break out of the loop
        
    return state

@njit()
def getAvailableMoves(state: np.array) -> np.array:
    """ Generates an array of elements set to 1 if the move is possible and 0 otherwise"""
    height, width = state.shape
    result = np.zeros((1, width)) # NOTE: Make sure that the dimensions of the output array matches 
    for i in range(width):
        if (state[0][i] == 0): result[0][i] = 1.0
    return result

def getStringRepresentation (state: np.array) -> str:
    """ Returns a string representation of the current state """
    lines = []
    for i in range(state.shape[0]):
        lines.append("\n|")
        for j in range(state.shape[1]):
            if (state[i, j] == 1.0):
                lines[-1] += "x|"
            elif (state[i, j] == -1.0):
                lines[-1] += "o|"
            else:
                lines[-1] += " |"
    return "".join(lines)