# An implementation of parallel MCTS
from typing import List, Tuple
import numpy as np 
from model import Net, device
from state import checkIfGameIsWon, generateEmptyState, getStringRepresentation, makeMove, validMoves
from config import Cpuct, enableValueHeadAfterIteration, rooloutsDuringTraining, width, height, mctsGPU, numberOfMapsPerPlayer, epsilon, disableValueHead
import torch
from logger import *
from utils import loadLatestModel
from predictor import Predictor

class ParallelMCTS:

    def __init__(self, iteration: int = 0, model: Net = None):
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
    
    def getActionProbs (self, states: List[np.array], roolouts: int, temperature: float = 1.0) -> List[np.array]:
        """
            Args:
                - State: The state of the game
                - Roolouts: the number of roolouts during search
                - Temperature: Determines how deterministically the moves are chosen.
            Returns:
                - A 3d Tensor in the dimensions (numberOfStates, 1, width)
                  containing the probabilities of picking each move in each state.
        """
        numberOfStates = len(states)
        
        # Populate the dictionaries 
        for _ in range(roolouts):
            self.search(states)
        
        counts = np.zeros((numberOfStates, 1, width))
        for idx, state in enumerate(states):
            s = state.tobytes() # This is used as the key
            counts[idx] = np.array([[self.Nsa[(s, a[1])] if ((s, a[1]) in self.Nsa) else 0 
                                for a, mask in np.ndenumerate(validMoves(state))]], dtype = "float32")
        
        probs = np.zeros((numberOfStates, 1, width), dtype = "float32")
        for idx, state in enumerate(states):
            if (temperature == 0.0):
                # If there is two instances of the max value this always picks the first one.
                probs[idx][0][np.argmax(counts[idx])] = 1.0 
        
            else:
                # Else compute a probability distribution on the moves based on the number of visits to each move
                probs[idx] = np.power(counts[idx], 1.0 / temperature) / np.sum(counts[idx])

        return probs
            
    def getMove (self, states: List[np.array], roolouts: int) -> List[int]:
        """ Returns the moves recommended by mcts (with temperature = 0.0) """
        return [np.argmax(self.getActionProbs(state, roolouts, temperature = 0.0)) for state in states]

    def search (self, states: List[np.array]): # TODO: This needs to be improved
        """ Performs the actual MCTS search recursively """
        # This is used as the keys in the dictionaries.
        byteStrings = [state.tobytes() for state in states] 
        numberOfStates = len(states)
        
        values, done = np.zeros(numberOfStates), [False for _ in range(numberOfStates)]     
        statesForPredictions, byteStringsForPredictions = [], set() # TODO: Turn this into a set to avoid duplicates
        for idx, (state, s) in enumerate(zip(states, byteStrings)):
            # Save board termination 
            if (s not in self.Es):
                self.Es[s] = checkIfGameIsWon(state)
                
            if (self.Es[s] != -1): # This is a terminal node 
                values[idx] = self.Es[s] # return 1 if the previous player won the game (0 otherwise)
                done[idx] = True
                
            elif (s not in self.Ps): # We have hit a leaf node
                # If the state has already been reached before (before the latest reset, it's still a leaf node after all) 
                if (s not in self.predictions): 
                    n = len(byteStringsForPredictions)
                    byteStringsForPredictions.add(s)
                    if (n != len(byteStringsForPredictions)):
                        statesForPredictions.append(state)
                    # done[idx] = True # We expanded the leaf node so we are done with this roolout
            
            elif (s not in self.predictions):
                n = len(byteStringsForPredictions)
                byteStringsForPredictions.add(s)
                if (n != len(byteStringsForPredictions)):
                    statesForPredictions.append(state)
                
        # Perform the predictions
        n = len(statesForPredictions)
        if (n != 0):
            predictions = self.predictor.predict(np.stack(statesForPredictions), n)
            for i in range(n):
                self.predictions[statesForPredictions[i].tobytes()] = (predictions[0][i], predictions[1][i][0])
        
        for idx, (state, s) in enumerate(zip(states, byteStrings)):
            if (done[idx] == False):
                self.Ps[s], v = self.predictions[s]
                    
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
                
                if (disableValueHead == False and self.iteration > enableValueHeadAfterIteration):
                    # This is in relation to the current player, 
                    # therefore the value that should be backpropagated is -v
                    values[idx] = -v 
                    done[idx] = True
                    
        if (all(done)): return values
        
        else:
            stateActionPairs = []
            for idx in range(numberOfStates):
                if (done[idx] == False):    
                    # Pick the action with the heighest UCBscore
                    best = {"score": -np.inf, "a": None}
                    s = byteStrings[idx]
                    
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
                    
                    stateActionPairs.append(((states[idx], s), best["a"]))
            
            # Compute the value recursively & and assign the values
            computedValues = self.search([makeMove(state, action) for (state, _), action in stateActionPairs])
            i = 0
            for idx in range(numberOfStates):
                if (done[idx] == False):
                    values[idx] = -computedValues[i]
                    i += 1            
            
            for (_, s), action in stateActionPairs:
                if (s, action) in self.Qsa:
                    self.Qsa[(s, action)] = (self.Nsa[(s, action)] * self.Qsa[(s, action)] + v) / (self.Nsa[(s, action)] + 1)
                    self.Nsa[(s, action)] += 1
                    
                else:
                    self.Qsa[(s, action)] = v
                    self.Nsa[(s, action)] = 1
                        
                self.Ns[s] += 1   
            
            return values
    
    def reset (self) -> None:
        """ Resets the dictionaries (excluding the predictions dictionary) """
        self.Qsa = {}; self.Nsa = {}; self.Ns = {}; 
        self.Es = {}; self.Ps = {}; self.Vs = {}
    
    def hardReset (self):
        """ Hard resets the dictionaries of the class """
        self.predictions = {}
        self.reset()
