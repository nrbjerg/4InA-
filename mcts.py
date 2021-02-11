from typing import Tuple
import numpy as np 
from model import Net, device
from state import checkIfGameIsWon, flipPlayer, generateEmptyState, makeMove, validMoves
from config import Cpuct, enableValueHeadAfterIteration, rooloutsDuringTraining, width, height, mctsGPU, numberOfMapsPerPlayer, epsilon, disableValueHead
import torch
from logger import *

class MCTS:
    
    def __init__ (self, model: Net, iteration: int = 0):
        """ Stores the data used for Monte-Carlo tree search """
        self.model = model
        self.model.eval()
        
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
                - Temperature: Determines how deterministicly the moves are chosen.
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
            return probs
        
        else:
            probs = np.power(counts, 1.0 / temperature) / np.sum(counts)
            return probs
    
    def getMove (self, state: np.array, roolouts: int) -> int:
        """ Returns the move recommended by the mcts """
        return np.argmax(self.getActionProbs(state, roolouts, temperature = 0.0))
    
    def predict (self, state: np.array, s: str) -> Tuple[np.array, float]:
        """ 
            Args:
                - State: The state which should be passed through the network 
                - s: The bytestring of the state, used for storing the models predictions in the predicions dictionary
            Returns:
                - A tuple containing the predicted policy a (1 x width) matrix 
                  & scalar containing the predicted value of the state.
        """
        with torch.no_grad():
            # Format the data correctly for the neural network & move it to gpu
            stateTensor = torch.from_numpy(state).view(1, 2 * numberOfMapsPerPlayer + 1, height, width)
            if (mctsGPU == True): 
                stateTensor = stateTensor.to(device)
    
            probs, value = self.model(stateTensor)

            # Move predictions back to cpu
            if (mctsGPU == True):
                probs = probs.cpu()
                value = value.cpu()
            
            # Stores the predictions of the model if none is stored already
            self.predictions[s] = (probs.numpy(), value[0][0])   
            
            return (probs.numpy(), value[0][0])
            
    def search (self, state: np.array):
        """ Performs the actual MCTS search recursively """
        s = state.tobytes() # This is used as the key
        
        # Save board termination 
        if (s not in self.Es):
            self.Es[s] = checkIfGameIsWon(state)
              
        if (self.Es[s] != -1): # This is a terminal node 
            return self.Es[s] # return 1 if the previous player won the game (0 otherwise)
        
        if (s not in self.Ps): # We have hit a leaf node
            
            # If the state has already been reached before (before the latest reset)
            if (s in self.predictions): 
                self.Ps[s], v = self.predictions[s]
            else:
                self.Ps[s], v = self.predict(state, s)
                
            valids = validMoves(state)
            self.Ps[s] = self.Ps[s] * valids # Mask probs (set probs[i] to 0 if i is not a valid move.)
            sumOfProbs = np.sum(self.Ps[s])
            
            if (sumOfProbs > 0): # Normalize the probs
                self.Ps[s] /= sumOfProbs
            else:
                # If all probs where but the state isn't terminal, give all valid moves equal probability
                warning("Warning: all moves where masked... ")
                self.Ps[s] = valids / np.sum(valids)
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            if (disableValueHead == False and self.iteration > enableValueHeadAfterIteration):
                # This is in relation to the current player, 
                # therefore the value that should be backpropagated is -v
                return -v 
            else: 
                # Enable value head after a couple of iterations             
                return 0 
            
        # Pick the action with the heighest UCBscore
        best = {"score": -np.inf, "a": None}
        
        for a in range(width):
            if (self.Vs[s][0][a] == 1):
                prior = self.Ps[s][0][a]
                if (s, a) in self.Qsa:
                    # Compute the UCB score
                    u = self.Qsa[(s, a)] + Cpuct * prior * np.sqrt(self.Ns[s]) /  (1 + self.Nsa[(s, a)])
                
                else:
                    # When Qsa = 0
                    u = Cpuct * prior * np.sqrt(self.Ns[s] + epsilon)
                
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
        """ Resets the dictionaries (exluding the predictions dictionary) """
        self.Qsa = {}; self.Nsa = {}; self.Ns = {}; 
        self.Es = {}; self.Ps = {}; self.Vs = {}
    
    def hardReset (self):
        """ Hard resets the dictionaries of the class """
        self.predictions = {}
        self.reset()
        
if (__name__ == "__main__"):
    mcts = MCTS(Net())
    print(mcts.getActionProbs(generateEmptyState(), rooloutsDuringTraining))