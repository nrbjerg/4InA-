import numpy as np
import os
from numpy.core.defchararray import not_equal
import tensorflow as tf
from tensorflow import keras 
from random import shuffle

from MCTS import MCTS
from State import State

class Trainer:
    
    def __init__ (self, model):
        self.MCTS = MCTS(model)
        self.model = model
        
    def executeEpisode (self, MTCSiterations: int):
        dataset = []
        currentPlayer = 1
        S = State()
        
        while True:
            S.map *= -1
            
            root = self.MCTS.run(S, MTCSiterations)
            
            # Actual action probs
            probs = [0 for _ in range(9)] # All of the posible moves
            for k, v in root.children.items():
                probs[k] = v.visits
            
            probs /= np.sum(probs) # Fix the sum of probs to 1 before training
            dataset.append((S.map, currentPlayer, probs))
            
            # Play the action recommend by MCTS
            action = root.selectAction(0)
            S.playMove(action)
            reward = S.isWonBy(currentPlayer)
            
            if reward != None:
                ret = []
                for m, player, probs in dataset:
                    # Mark the player who won and the one who lost
                    if (player != currentPlayer):
                        ret.append(m, probs, reward * (-1))
                    else:
                        ret.append(m, probs, reward)
                
                return ret        
            currentPlayer *= -1
        
    def createDataset (self, MCTSiterations: int, episodes: int):
        """ Creates a set of training data for the algoritm to train upon """
        # Create the dataset
        dataset = []
        for _ in range(episodes):
            dataset.extend(self.executeEpisode(MCTSiterations))

        # Shuffle for better learning
        shuffle(dataset)
        return dataset
    
    def train (self, ):
        
    def saveCheckPoint (self, folder, filename):
        """ Saves the current model """
        if (not os.path.exists(folder)):
            os.mkdir(folder)
            
        filepath = os.path.join(os.getcwd(), folder, filename)
        # TODO: Implement the last part of this function
    