from mcts import getAction
from state import getAvailableMoves, getStringRepresentation, checkIfGameIsWon, makeMove
from model import Net, loadLatetestModel
import numpy as np

if (__name__ == "__main__"):
    model = loadLatetestModel()
    state = np.zeros((6, 7), dtype = "float32")
    state = -makeMove(state.copy(), getAction(state.copy(), 128, model))
    turn = 1
    
    while (checkIfGameIsWon(state) == -1):
        if ((turn % 2) == 0):
            # It's the neural nets turn 
            state = makeMove(state.copy(), getAction(state.copy(), 128, model))
        else:
            # It's your turn
            print(getStringRepresentation(state))
            idx = int(input("Move idx: "))
            if (getAvailableMoves(state)[0][idx] == 0):
                print("Move is not available")
            else:
                state = makeMove(state.copy(), idx)
            pass
        state *= -1
        turn += 1
    print(getStringRepresentation(state))