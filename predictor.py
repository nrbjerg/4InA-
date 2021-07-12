from model import Net, device
from config import mctsGPU, numberOfMapsPerPlayer, height, width
from typing import List, Tuple
import numpy as np
import torch
from utils import loadLatetestModel

class Predictor:
    
    def __init__ (self, model: Net):
        self.model = model
        self.model.eval()
        
        if (mctsGPU == True): self.model.cuda()
        
    def updateModel (self): 
        """ Load the latest model """
        self.model, _ = loadLatetestModel()
        self.model.eval()
        
        if (mctsGPU == True): self.model.cuda()
        
    def predict (self, states: List[np.array], numberOfStates: int) -> Tuple[np.array]:
        """ 
            Args:
                - states: The states which should be passed through the network 
                - byteStrings: The bytestring corresponding to the state, 
                  used for storing the models predictions in the predictions dictionary
        """
        
        with torch.no_grad():
            # Format the data correctly for the neural network & move it to gpu
            stateTensor = torch.from_numpy(states).view(numberOfStates, 2 * numberOfMapsPerPlayer + 1, height, width)
            if (mctsGPU == True): 
                stateTensor = stateTensor.to(device)
    
            probs, values = self.model(stateTensor)
            # Move predictions back to cpu
            if (mctsGPU == True):
                probs = probs.cpu().numpy()
                values = values.cpu().numpy()
            
            return (probs, values)
             
predictor = Predictor(loadLatetestModel()[0])