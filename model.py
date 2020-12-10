import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.keras import activations
from State import State
import os
import numpy as np
from State import State

class Model:
    
    def __init__ (self, version: int = None, n: int = 2):
        """ Loads a previus model if version != none, if version == none create an entierly new model """
        if version == None:
            self.valueNetwork = keras.models.Sequential()
            self.valueNetwork.add(tf.keras.layers.Flatten(input_shape = (3, 3)))
            self.valueNetwork.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
        else:
            pass # TODO: implement
    
    def predict (self, S: State) -> (np.array):
        """ Predicts the prior propablities and the value of the state """
        print(S.map.shape)
        return (self.valueNetwork.predict(S.map))

if __name__ == "__main__":
    S = State()
    M = Model(n = 1)
    print(M.predict(S))