from typing import Tuple, Union
from utils import loadLatestModel
import numpy as np 
from model import Net, device
from state import checkIfGameIsWon, flipPlayer, generateEmptyState, makeMove, validMoves
from config import Cpuct, enableValueHeadAfterIteration, rooloutsDuringTraining, width, height, mctsGPU, numberOfMapsPerPlayer, epsilon, disableValueHead
import torch
from logger import *
from predictor import Predictor

class MCTS:
    
    def __init__ (self, model: Net = None, iteration: int = 0):
        """ Stores the data used for Monte-Carlo tree search """
        self.predictor = Predictor(model or loadLatestModel()[0])
        self.iteration = iteration
        
        self.predictions = {} # Stores the predictions of the model
        
        # NOTE: Each state will be stored under the key: state.tobytes(), meaning that the byte string representing the numpy array will be used as a key
        self.Qsa = {} # The Q values for s,a
        self.Nsa = {} # The number of times the action a has been taken from state s
        self.Ns = {} # The number of times the state has been visited
        
        self.Ps = {} # Stores the initial policy (from the neural network)
        
        self.Es = {} # Stores if the current state has terminated (1 for win, 0 for draw, -1 for not terminated)
        self.Vs = {} # Stores the valid moves for the state s
        
    def getActionProbs(self, state: np.array, roolouts: int, temperature: float = 1.0) -> np.array:
        """
            Args:
                - State: The state of the game
                - Roolouts: the number of roolouts during search
                - Temperature: Determines how deterministically the moves are chosen.
            Returns:
                - A 2d matrix in the dimensions (1, width) containing the probabilities of picking each move.
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
        """ Returns the move recommended by the mcts """
        return np.argmax(self.getActionProbs(state, roolouts, temperature = 0.0))

    def computeUCBScore (self, action: int, byteString: str, prior: float) -> float:
        """ Computes the UCB score of a node based on the arguments """
        if (byteString, action) in self.Qsa: # if its in Qsa its in Nsa as well.
            # Compute the UCB score
            return self.Qsa[(byteString, action)] + Cpuct * prior * np.sqrt(self.Ns[byteString]) /  (1 + self.Nsa[(byteString, action)])
                
        # When Qsa = 0
        return Cpuct * prior * np.sqrt(self.Ns[byteString] + epsilon)

    def pickActionWithHighestUCBScore(self, byteString) -> int:
        """ Pick the action by idx with the heighest UCBscore """
        best = {"score": -np.inf, "a": None}
        
        for a in range(width): 
            if (self.Vs[byteString][0][a] == 1): # For each valid action ;)
                prior = self.Ps[byteString][0][a]
                u = self.computeUCBScore(a, byteString, prior)

                if (u > best["score"]):
                    best["score"] = u
                    best["a"] = a

        return best["a"]
 
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

    def search (self, state: np.array): 
        """ Performs the actual MCTS search recursively """
        byteString = state.tobytes() # This is used as the key
        
        # Save board termination 
        if (byteString not in self.Es):
            self.Es[byteString] = checkIfGameIsWon(state)
              
        if (self.Es[byteString] != -1): # This is a terminal node 
            return self.Es[byteString] # return 1 if the previous player won the game (0 otherwise)
        
        if (byteString not in self.Ps): # We have hit a leaf node
            
            # Perform prediction if needed & and store said prediction for later
            self.Ps[byteString], v = self.predict(state, byteString) 
                
            valids = validMoves(state)
            self.Ps[byteString] = self.Ps[byteString] * valids # Mask probs (set probs[i] to 0 if i is not a valid move.)
            sumOfProbs = np.sum(self.Ps[byteString])
            
            if (sumOfProbs > 0): # Normalize the probs
                self.Ps[byteString] /= sumOfProbs
            else:
                # If all probs where but the state isn't terminal, give all valid moves equal probability
                logWarning("Warning: all moves where masked... ")
                self.Ps[byteString] = valids / np.sum(valids)
            
            self.Vs[byteString] = valids
            self.Ns[byteString] = 0
            
            if (disableValueHead == False and self.iteration > enableValueHeadAfterIteration):
                # This is in relation to the current player, 
                # therefore the value that should be backpropagated is -v
                return -v 
            else: 
                # Enable value head after a couple of iterations             
                return 0 
            
        # Pick the action with the heighest UCBscore
        # Compute the value recursively
        action = self.pickActionWithHighestUCBScore(byteString)
        v = self.search(makeMove(state, action))
        
        if (byteString, action) in self.Qsa:
            self.Qsa[(byteString, action)] = (self.Nsa[(byteString, action)] * self.Qsa[(byteString, action)] + v) / (self.Nsa[(byteString, action)] + 1)
            self.Nsa[(byteString, action)] += 1
        else:
            self.Qsa[(byteString, action)] = v
            self.Nsa[(byteString, action)] = 1
                    
        self.Ns[byteString] += 1   
        
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
    mcts = MCTS(Net())
    print(mcts.getActionProbs(generateEmptyState(), rooloutsDuringTraining))