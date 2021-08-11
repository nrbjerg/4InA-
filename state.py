import numpy as np
from numba import njit, prange
from config import width, height, numberOfMapsPerPlayer

@njit()
def flipPlayer (state: np.array):
    """ Flips the values in the last index of the states tensor from 0 to 1 or from 1 to 0 """
    state[-1].fill(1 if state[-1][0][0] != 1 else 0) # TODO: Check if this should be -1

@njit()
def generateEmptyState () -> np.array:
    """ 
        Returns:
            - An empty state with no moves played
    """
    state = np.zeros((2 * numberOfMapsPerPlayer + 1, height, width), dtype = np.float32)
    state[-1].fill(1) # Indicate that it's the turn of player x
    return state

def flipGameState (state: np.array) -> np.array:
    """ Flips the board around the y axis & returns a new view """
    return np.flip(state, axis = 2)
    
@njit()
def checkIfGameIsWon(state: np.array) -> int:
    """ 
        Checks if the last player won the game, given the current state of the game
        Returns:
            1  if the last player won the game.
            0  if the game was a draw.
            -1 if the game hasn't ended
    """
    playerIdx = 0 if (state[-1][0][0] == -1) else numberOfMapsPerPlayer

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

    # NOTE: Numba doesn't support this kind of program otherwise this would be cleaner (but slower)
    # if all(mask[0][i] != 0.0 for i in range(width)):
    #     return -1

    # 5. The game is a draw if there is no empty moves
    return 0
    
@njit()
def makeMove (state: np.array, idx: int) -> np.array:
    """ 
        Args: 
            - State: the numpy array currently representing the game
            - Idx: the index of the move (along the x axis)
        Returns:
            - A copy of the state, with the new move played. (Also the player is switched)
    """
    currentMapIdx = 0 if (state[-1][0][0] == 1.0) else numberOfMapsPerPlayer
    state = state.copy()
    
    # Play moves from the button up
    for i in range(1, height + 1):
        if (state[0][height - i][idx] == 0.0 and state[numberOfMapsPerPlayer][height - i][idx] == 0.0):
            # Play the move
            state[currentMapIdx][height - i][idx] = 1.0
            break
    
    flipPlayer(state)
    return state

@njit()
def validMoves(state: np.array) -> np.array:
    """ 
        Args: 
            - State: the numpy array currently representing the game
        Returns:
            - A row vector with each element set to 0 
              if the appropriate move is not valid and 1 otherwise.
    """
    result = np.zeros((1, width), dtype = np.float32) 
    for i in range(width):
        if (state[0][0][i] == 0.0 and state[numberOfMapsPerPlayer][0][i] == 0.0): result[0][i] = 1.0
    return result

def getStringRepresentation(state: np.array) -> str:
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
    return "".join(lines) + "\n" + "".join(" " + str(i) for i in range(width)) + "\n"

if (__name__ == "__main__"):
    s = generateEmptyState()
    s = makeMove(s, 1)
    # s = makeMove(s, 2)
    print(s)
