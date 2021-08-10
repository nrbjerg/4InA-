from state import *
from mcts import MCTS
from utils import loadLatestModel
from model import Net
from config import rooloutsDuringTraining

state = generateEmptyState()
mcts = MCTS(loadLatestModel()[0])

while (checkIfGameIsWon(state) == -1):
    print(getStringRepresentation(state), "\n")
    if (state[-1][0][0] == 0):
        valids = validMoves(state)
        try:
            move = int(input("Your move index: "))
        except ValueError:
            move = int(input("Please specify a move index: "))
        while (valids[0][move] != 1):
            move = int(input("Your input was invalid, please input new move: "))
        state = makeMove(state, move)
    else:
        state = makeMove(state, mcts.getMove(state, rooloutsDuringTraining))
    
    print(getStringRepresentation(state))