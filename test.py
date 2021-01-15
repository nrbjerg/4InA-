from mcts import getAction
from state import getAvailableMoves, getStringRepresentation, checkIfGameIsWon, makeMove
from model import Net, loadLatetestModel, loadModel
from train import playGame, evaluateModel

model = loadModel("0.pt")
print(f"Trained model against the first model: {-evaluateModel(model, mctsSimulations = 50)}")