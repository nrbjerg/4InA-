from typing import Dict, List, Tuple
from evaluate import evaluateModel, playGame
from utils import fileNumbersInDirectory, loadModel
import numpy as np 
import json 
from numpy.random import choice
from mcts import MCTS
from state import makeMove, generateEmptyState
from config import width
from tqdm import tqdm
from copy import deepcopy

def loadEloes () -> Dict[str, Dict[str, int]]:
    """ Load the eloes from the eloes.json file in the data directory """
    try:
        with open("data/eloes.json", "r") as jsonFile:
            return json.load(jsonFile)
    except FileNotFoundError:
        
        dictionary = {}
        for n in fileNumbersInDirectory("models"):
            dictionary[n] = {"p+v": 1000, "p": 1000} # PV = Policy + Value, P = Policy
        dictionary["iteration"] = 0
        return dictionary

def getModelPairs (models: List[str]) -> List[Tuple[str]]:
    """ Pair each model against a random other model and return the pairs as a list of tuples """
    if ((len(models) % 2) != 0):
        raise IndexError("There have to be an equal number of models")
    else:
        pairs = []
        
        while (len(models) > 0):
            pair = []
            
            for _ in range(2):
                pair.append(choice(models))
                models.remove(pair[-1]) # Remove already chosen models from the pool
            
            pairs.append(tuple(pair))
            
        return pairs
    
def computeEloes (iterations: int):
    """ Each model plays 2 games per iteration afterwards the elo of each model is updated.  """
    eloes = loadEloes()
    models = fileNumbersInDirectory("models")
    
    agents = {}
    for m in models:
        agents[m] = {}
        agents[m]["p+v"] = MCTS(loadModel(m + ".pt"), iteration = np.inf)
        agents[m]["p"] = MCTS(loadModel(m + ".pt"), iteration = 0)
    
    for iteration in range(eloes["iteration"] + 1, iterations):
        print(f"Currently at elo iteration: {iteration}")
        eloes = loadEloes()
        eloes["iteration"] = iteration # Update the current iteration data 
        
        # Each model plays a single game afterwards the elo of each model is updated
        pairs = getModelPairs(deepcopy(models))
        
        for p in tqdm(pairs):
            for key in ["p+v", "p"]:
                # Generate a random starting gamestate.  
                s = makeMove(generateEmptyState(), np.random.randint(0, width))
                
                wins, losses = playGame(s.copy(), agents[p[0]][key], agents[p[1]][key], 0, 0)
                losses, wins = playGame(s.copy(), agents[p[1]][key], agents[p[0]][key], losses, wins)
                
                # Expected probabilities that either player wins 
                probs = [1 / (1 + 10**((eloes[p[1]][key] - eloes[p[0]][key]) / 400)),
                         1 / (1 + 10**((eloes[p[0]][key] - eloes[p[1]][key]) / 400))]
                
                if (wins > losses): # Model p[0] won
                    eloes[p[0]][key] += 32 * (1.0 - probs[0])
                    eloes[p[1]][key] += 32 * (0.0 - probs[1])
                    
                elif (wins < losses): # Model p[1] won
                    eloes[p[0]][key] += 32 * (0.0 - probs[0])
                    eloes[p[1]][key] += 32 * (1.0 - probs[1])
                 
                else: # p[0] and p[1] drew the game
                    eloes[p[0]][key] += 32 * (0.5 - probs[0])
                    eloes[p[1]][key] += 32 * (0.5 - probs[1])
                
                eloes[p[0]][key] = round(eloes[p[0]][key])
                eloes[p[1]][key] = round(eloes[p[1]][key])
                
        # Save new elo data, then the script can be terminated without worry
        with open("data/eloes.json", "w") as file:
            json.dump(eloes, file)
            
        
if (__name__ == "__main__"):
    computeEloes(100)