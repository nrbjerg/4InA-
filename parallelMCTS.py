from typing import List, Tuple
import numpy as np 
from model import Net, device
from state import checkIfGameIsWon, generateEmptyState, makeMove, validMoves
from config import Cpuct, enableValueHeadAfterIteration, rooloutsDuringTraining, width, height, mctsGPU, numberOfMapsPerPlayer, epsilon, disableValueHead
import torch
from logger import *

class ParallelMCTS:
    
    def __init__ (self, model: Net, iteration: int = 0):
        """ 
            Stores the data used for Monte-Carlo tree search 
            this class uses monte carlo tree search on multiple games at once,
            this increases GPU saturation and improves performance during training.
        """
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
        
    def getActionProbs (self, states: List[np.array], roolouts: int, temperature: float = 1.0) -> List[np.array]:
        """
            Args:
                - States: The state of the game
                - Roolouts: the number of roolouts during search
                - Temperature: Determines how deterministicly the moves are chosen.
            Returns:
                - A list of 2d matrices in the dimensions (1, width) containing the probabilities of picking each move.
        """
        # Populate the dictionaries 
        for _ in range(roolouts):
            self.search(states) # TODO: Rewrite search
        
        byteStrings = [s.tobytes() for s in states] # This is used as the keys
        
        probs = []
        for byteString, state in zip(byteStrings, states):
            counts = np.array([[self.Nsa[(byteString, a[1])] if ((byteString, a[1]) in self.Nsa) else 0 
                                for a, mask in np.ndenumerate(validMoves(state))]], dtype = "float32")
            
            if (temperature == 0.0):
                probs.append(np.zeros((1, width), dtype = "float32"))
                # NOTE: If there is two instances of the max value this always picks the first one.
                probs[-1][0][np.argmax(counts)] = 1.0 
            
            else:
                probs.append(np.power(counts, 1.0 / temperature) / np.sum(counts))
    
        return probs
    
    def getMove (self, states: List[np.array], roolouts: int) -> int:
        """ Returns the moves recommended by mcts (with temperature = 0.0) """
        return [np.argmax(p) for p in self.getActionProbs(states, roolouts, temperature = 0.0)]
    
    def predict (self, states: List[np.array], byteStrings: List[str]) -> None:
        """ 
            Args:
                - States: The states which should be passed through the network 
                - byteStrings: The byteStrings of the states, used for storing the models predictions in the predictions dictionary
        """
        with torch.no_grad():
            # Format the data correctly for the neural network & move it to gpu
            stateTensor = torch.from_numpy(np.stack(states)).view(len(states), 2 * numberOfMapsPerPlayer + 1, height, width)
            if (mctsGPU == True): 
                stateTensor = stateTensor.to(device)
    
            probs, values = self.model(stateTensor)

            # Move predictions back to cpu
            if (mctsGPU == True):
                probs = probs.cpu()
                value = values.cpu()
            
            # Stores the predictions of the model if none is stored already
            for byteString, p, v in zip(byteStrings, probs, values):
                self.Ps[byteString] = p.numpy()
                self.V = v[0][0]
            
    def search (self, states: List[np.array]) -> List[float]:
        """ Performs the actual MCTS search recursively """
        byteStrings = [s.tobytes() for s in states] # These will be used as keys
        
        # Compute predictions together
        statesForPrediction, byteStringsForPredictions = []
        
        values = np.zeros(7, dtype = "float32")

        for idx, (byteString, state) in enumerate(zip(byteStrings, states)):
            # Save board termination 
            if (byteString not in self.Es):
                self.Es[byteString] = checkIfGameIsWon(state)
                
            if (self.Es[byteString] != -1): # This is a terminal node 
                values[idx] = self.Es[byteString] # return 1 if the previous player won the game (0 otherwise)
            
            if (byteString not in self.Ps): # We have hit a leaf node
                
                # If the state has already been reached before (before the latest reset) 
                if (byteString not in self.V): 
                    statesForPrediction.append((state, byteString))
            
        for byteString, state in zip(byteStrings, states):
            
            valids = validMoves(state)
            self.Ps[byteString] = self.Ps[byteString] * valids # Mask probs (set probs[i] to 0 if i is not a valid move.)
            sumOfProbs = np.sum(self.Ps[byteString])
            
            if (sumOfProbs > 0): # Normalize the probs
                self.Ps[byteString] /= sumOfProbs
            else:
                # If all probs where but the state isn't terminal, give all valid moves equal probability
                warning("Warning: all moves where masked... ")
                self.Ps[byteString] = valids / np.sum(valids)
            
            self.Vs[byteString] = valids
            self.Ns[byteString] = 0
            
            if (disableValueHead == False and self.iteration > enableValueHeadAfterIteration):
                # This is in relation to the current player, 
                # therefore the value that should be backpropagated is -v
                return -self.V[byteString] 
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