from typing import List, Tuple, Union
import numpy as np 
from model import Net, device
from state import checkIfGameIsWon, generateEmptyState, makeMove, validMoves
from config import Cpuct, enableValueHeadAfterIteration, rooloutsDuringTraining, width, height, mctsGPU, numberOfMapsPerPlayer, epsilon, disableValueHead
import torch
from logger import *
from predictor import Predictor
from utils import loadLatestModel

class MCTS:
    
    def __init__ (self, iteration: int = 0, model: Net = None):
        """ Stores the data used for Monte-Carlo tree search """
        self.predictor = Predictor(model or loadLatestModel()[0])
        
        if (mctsGPU == True): self.model.cuda()
        
        self.iteration = iteration
        self.predictions = {} # Stores the predictions of the model
        
        # NOTE: Each state will be stored under the key: state.tobytes(), meaning that the byte string representing the numpy array will be used as a key
        self.Qsa = {} # The Q values for s,a
        self.Nsa = {} # The number of times the action a has been taken from state s
        self.Ns = {} # The number of times the state has been visited
        
        self.Ps = {} # Stores the initial policy (from the neural network)
        
        self.Es = {} # Stores if the current state has terminated (1 for win, 0 for draw, -1 for not terminated)
        self.Vs = {} # Stores the valid moves for the state s
        
    def getActionProbs (self, state: np.array, roolouts: int, temperature: float = 1.0) -> np.array:
        """
            Args:
                - State: The state of the game
                - Roolouts: the number of roolouts during search
                - Temperature: Determines how deterministically the moves are chosen.
            Returns:
                - A row vector containing the probabilities of picking each move.
        """
        # Populate the dictionaries 
        for _ in range(roolouts):
            self.search(state)

        s = state.tobytes() # This is used as the key
        counts = np.array([[self.Nsa[(s, a[1])] if ((s, a[1]) in self.Nsa) else 0 
                            for a, mask in np.ndenumerate(validMoves(state))]], dtype = "float32")

        if (temperature == 0.0):
            probs = np.zeros((1, width), dtype = "float32")
            # If there is two instances of the max value this always picks the first one.
            probs[0][np.argmax(counts)] = 1.0
        else:
            probs = np.power(counts, 1.0 / temperature) / np.sum(counts)

        return probs
    
    def getMove (self, state: np.array, roolouts: int) -> int:
        """ Returns the moves recommended by mcts (with temperature = 0.0) """
        return np.argmax(self.getActionProbs(state, roolouts, temperature = 0.0))
    
    def predict (self, state: np.array, byteString: str) -> Tuple[Union[np.array, int]]:
        """ Performs the actual prediction if needed (ie. if its not already stored in the predictions dictionary) """
        # If the state has already been reached before the latest reset of the class, 
        # then we don't need to perform the actual predictions with our model 
        if (byteString in self.predictions) == False:
            modelPrediction = self.predictor.predict(state, numberOfStates = 1)
            # NOTE: The predictor actually returns a tuple of numpy arrays in the dimensions j of predictions 
            # (in this case (np.array of size (numberOfStates, 7), np.array of size (numberOfStates, 1)
            self.predictions[byteString] = (modelPrediction[0][0], modelPrediction[1][0][0]) 

        return self.predictions[byteString]

    def computeUCBScore (self, action: int, byteString: str, prior: float) -> float:
        """ Computes the UCB score of a node based on the arguments """
        if (byteString, action) in self.Qsa: # if its in Qsa its in Nsa as well.
            # Compute the UCB score
            return self.Qsa[(byteString, action)] + Cpuct * prior * np.sqrt(self.Ns[byteString]) /  (1 + self.Nsa[(byteString, action)])
                
        # When Qsa = 0
        return Cpuct * prior * np.sqrt(self.Ns[byteString] + epsilon)
 
    def search (self, state: np.array): # TODO: Split into smaller methods
        """ Performs the actual MCTS search recursively """
        s = state.tobytes() # This is used as the key

        # Save board termination (-1 if the state isn't a terminal state)
        if (s not in self.Es):
            self.Es[s] = checkIfGameIsWon(state)
              
        if (self.Es[s] != -1): # This is a terminal node 
            return self.Es[s] # return 1 if the previous player won the game (0 otherwise)
        
        if (s not in self.Ps): # We have hit a leaf node
            # Let the model predict the actual probabilities ect.
            self.Ps[s], v = self.predict(state, s) 

            valids = validMoves(state)
            self.Ps[s] = self.Ps[s] * valids # Mask probs (set probs[i] to 0 if i is not a valid move.)
            sumOfProbs = np.sum(self.Ps[s])
            
            if (sumOfProbs > 0): # Normalize the probs
                self.Ps[s] /= sumOfProbs
            else:
                # If all probs where 0 but the state isn't terminal, give all valid moves equal probability
                warning("Warning: all moves where masked... ")
                self.Ps[s] = valids / np.sum(valids)
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            if (disableValueHead == False or self.iteration > enableValueHeadAfterIteration):
                # This is in relation to the current player, 
                # therefore the value that should be backpropagated is -v
                return -v 
            else: 
                # Enable value head after a couple of iterations             
                return 0 
            
        # Pick the action with the heighest UCBscore
        best = {"score": -np.inf, "a": None}
        
        for a in range(width): # For each action
            if (self.Vs[s][0][a] == 1):

                prior = self.Ps[s][0][a]
                u = self.computeUCBScore(a, s, prior)

                if (u > best["score"]):
                    best["score"] = u
                    best["a"] = a
        
        # Compute the value recursively
        action = best["a"]
        v = self.search(makeMove(state, action))
        
        if (s, action) in self.Qsa:
            self.Qsa[(s, action)] = (self.Nsa[(s, action)] * self.Qsa[(s, action)] + v) / (self.Nsa[(s, action)] + 1)
            self.Nsa[(s, action)] += 1
            
        else:
            self.Qsa[(s, action)] = v
            self.Nsa[(s, action)] = 1
                    
        self.Ns[s] += 1   
        
        return -v
    
    def reset (self) -> None:
        """ Resets the dictionaries (excluding the predictions dictionary) """
        self.Qsa = {}; self.Nsa = {}; self.Ns = {}; 
        self.Es = {}; self.Ps = {}; self.Vs = {}
    
    def hardReset (self):
        """ Hard resets the dictionaries of the class """
        self.predictions = {}
        self.reset()
        
if (__name__ == "__main__"):
    mcts = MCTS(model = Net())
    print(mcts.getActionProbs(generateEmptyState(), roolouts = 10))
