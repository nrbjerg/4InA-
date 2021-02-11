import json 
from evaluate import evaluateModel
from utils import loadLatetestModel
from utils import fileNumbersInDirectory, loadModel
import numpy as np
from tqdm import tqdm 

def computeWinratesAgainstStartingModel ():
    """ Computes the winrate of each model against the starting model. """
    # Load the first model
    startingModel = loadModel("0.pt")
    dictionary = {}
    
    # Compute the winrate of each model
    for model in sorted(fileNumbersInDirectory("models")):
        print(f"At model: {model}")
        dictionary[str(model)] = {}
        dictionary[str(model)]["p+v"] = evaluateModel(loadModel(str(model) + ".pt"), np.inf, opponent = startingModel)
        dictionary[str(model)]["p"] = evaluateModel(loadModel(str(model) + ".pt"), 0, opponent = startingModel)

    # Save the winrates
    with open("data/winrates.json", "w") as file:
        json.dump(dictionary, file)
        
if (__name__ == "__main__"):
    computeWinratesAgainstStartingModel()