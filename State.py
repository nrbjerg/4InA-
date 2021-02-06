import numpy as np
from numba import njit, prange
from config import width, height, numberOfMapsPerPlayer

@njit()
def flipPlayer (state: np.array):
    """ Sets the last idx of state to match the current player """
    state[-1].fill(1 if state[-1][0][0] != 1 else 0)

@njit
def generateEmptyState () -> np.array:
    """ Generates an empty state """
    # State[:numberOfMapsPerPlayer] = x, State[numberOfMapsPerPlayer:2 * numberOfMapsPerPlayer] = o, state[-1] = indicates the current player
    state = np.zeros((2 * numberOfMapsPerPlayer + 1, height, width), dtype = np.float32)
    state[-1].fill(1)
    return state

def flipGameState (state: np.array) -> np.array:
    """ Flips the board around the y axis """
    return np.flip(state, axis = 2)
    
@njit()
def checkIfGameIsWon (state: np.array) -> int:
    """ 
        Checks if the last player won the game
        Returns:
            1  if the last player won the game.
            0  if the game was a draw.
            -1 if the game hasn't ended
    """
    playerIdx = 0 if (state[-1][0][0] == 0.0) else numberOfMapsPerPlayer

    s = state[playerIdx]
        
    # 1. Check for horizontal alignments
    for i in prange(height):
        for j in range(width - 3):
            if (s[i, j] == 1.0 and s[i, j + 1] == 1.0 and s[i, j + 2] == 1.0 and s[i, j + 3] == 1.0): return 1
                
    # 2. Check for vertical alignments
    for i in range(height - 3):
        for j in range(width):
            if (s[i, j] == 1.0 and s[i + 1, j] == 1.0 and s[i + 2, j] == 1.0 and s[i + 3, j] == 1.0): return 1
     
    # 3. Check for diagonal alignments
    for i in range(height - 3):    
        for j in range(width - 3):
            if (s[i, j] + s[i + 1, j + 1] + s[i + 2, j + 2] + s[i + 3, j + 3] == 4.0): return 1
            if (s[i, j + 3] + s[i + 1, j + 2] + s[i + 2, j + 1] + s[i + 3, j] == 4.0): return 1
    
    # 4. Check for empty moves
    mask = state[0] + state[numberOfMapsPerPlayer]
    for i in range(width):
        if (mask[0][i] == 0.0): return -1
    
    # 5. The game is a draw if there is no empty moves
    return 0
    
@njit()
def makeMove (state: np.array, idx: int) -> np.array:
    """ Copies the state and makes a move on the copy """
    # TODO: Add support for history
    mask = state[0] + state[numberOfMapsPerPlayer]
    currentMapIdx = 0 if (state[-1][0][0] == 1.0) else numberOfMapsPerPlayer
    state = state.copy()
    for i in range(1, height + 1): # Play moves from the button up
        if (mask[height - i][idx] == 0.0):
            # Play the move
            state[currentMapIdx][height - i][idx] = 1.0
            break
    
    # Flips the player
    flipPlayer(state) 
       
    return state

@njit()
def validMoves(state: np.array) -> np.array:
    """ Generates an array of elements set to 1 if the move is possible and 0 otherwise"""
    mask = state[0] + state[numberOfMapsPerPlayer]
    result = np.zeros((1, width), dtype = np.float32) # NOTE: Make sure that the dimensions of the output array matches 
    for i in range(width):
        if (mask[0][i] == 0.0): result[0][i] = 1.0
    return result

def getStringRepresentation (state: np.array) -> str:
    """ Returns a string representation of the current state """
    x, o = state[0], state[numberOfMapsPerPlayer]
    lines = []
    for i in range(height):
        lines.append("\n|")
        for j in range(width):
            if (x[i, j] == 1.0):
                lines[-1] += "x|"
            elif (o[i, j] == 1.0):
                lines[-1] += "o|"
            else:
                lines[-1] += " |"
    return "".join(lines)

if __name__ == '__main__':
    state = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
                      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                       [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]],
                      [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
    # print(checkIfGameIsWon(state))
    print(getStringRepresentation(state))
    print(getStringRepresentation(flipGameState(state)))
    
