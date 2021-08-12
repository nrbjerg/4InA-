from typing import List
import os 
import numpy as np
from utils import loadModel
from model import Net
from evaluate import evaluateModel
import matplotlib.pyplot as plt

def computeWinrate (modelIteration: int, initialModel: Net) -> float:
	""" Computes the winrate of the model against the initial model """
	model = loadModel(f"{modelIteration}.pt")
	
	return evaluateModel(model, np.inf, opponent = initialModel) # np.inf is so that the value head is enabled (its enabled in MCTS search if iteration > enableValueHeadAfterIteration)

def computeWinrates(modelIterations: List[int]) -> List[float]:
	""" Computes the winrate of each model against the initial model and returns these winrates in the form of a list """
	initialModel = loadModel("0.pt")
	return [computeWinrate(modelIteration, initialModel)
	    	for modelIteration in modelIterations]

def visualizeWinRates (modelIterations: List[int]) -> List[float]:
	""" Visualizes the winrates against the initial model """ 
	winrates = computeWinrates(modelIterations)
	plt.style.use("ggplot")
	plt.title("winrate against initial model")
	plt.ylabel("winrate in %")
	plt.xlabel("model iteration")
	plt.plot(modelIterations, winrates)
	plt.show()

if (__name__ == "__main__"):
    files = os.listdir(os.path.join(os.getcwd(), "models"))
    iterations = sorted([int(f.split(".")[0]) for f in files])
    visualizeWinRates(iterations)