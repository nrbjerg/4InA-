from torch.types import Number
from utils import loadLatetestModel, saveModel
from evaluate import evaluateModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from model import Net
from config import learningRate, disableValueHead, batchSize, epochs, iterations, trainingOnGPU, mctsGPU, window
import numpy as np
from tqdm import trange
from logger import info, error
from data import createDataset, convertTrainingDataToTensors, saveDataset, loadDataset

mse = nn.MSELoss()

def crossEntropy (predictions: Tensor, targets: Tensor) -> float:
    return -torch.sum(targets * F.log(predictions))

def updateWeights (model: Net, states: torch.Tensor, probs: torch.Tensor, rewards: torch.Tensor) -> Net:
    """ 
        Updates the weights of the neural network based on the training data 
            x: states,
            ys: policies and rewards 
    """
    print("Updating weights:")
    # 1.1 Set model to training mode 
    model.train()
    
    # 1.2 Move model to gpu or back to cpu
    if (mctsGPU == False and trainingOnGPU == True): 
        model.cuda()
        
    elif (mctsGPU == True and trainingOnGPU == False):
        model.cpu()
    
    # 2. Train the neural network
    with trange(epochs, unit = "epoch") as t:
        for e in t:
            optimizer = Adam(model.parameters(), lr = learningRate)  
            
            permutation = torch.randperm(states.size()[0])
            
            valueLoss = 0
            policyLoss = 0
            batches = 0
            for idx in range(0, states.size()[0], batchSize):
                batches += 1
                # Load minibatch traning data
                indices = permutation[idx:idx + batchSize]
                s = states[indices]
                p = probs[indices]
                r = rewards[indices]
                
                # Pass data though network & update weights using the optimizer
                optimizer.zero_grad()
                predictions = model(s)
                
                # Check outputs for NaN
                if (torch.sum(predictions[0] != predictions[0])): error("nan in policy")
                elif (torch.sum(predictions[1] != predictions[1])): error("nan in value")
                
                policyLoss = mse(predictions[0], p) # TODO: Update this to another loss function
                valueLoss = 0
                if (disableValueHead == False):
                    valueLoss = mse(predictions[1], r)
                    loss = policyLoss + valueLoss
                else: 
                    loss = policyLoss
                
                loss.backward()
                optimizer.step()
                
                # 2.3 Update the total loss
                policyLoss += policyLoss.item()
                if (disableValueHead == False):
                    valueLoss += valueLoss.item()

            policyLoss /= batches
            valueLoss /= batches
            
            # Update progres bar
            if (disableValueHead == False):
                t.set_postfix(pl = policyLoss.item(), vl = valueLoss.item())
            else: 
                t.set_postfix(pl = policyLoss.item())
            
            # Log information about the losses
            if (e == epochs  - 1):
                if (disableValueHead == False):
                    info(f"Ended with losses: \n - value loss: {valueLoss.item():.4f}\n - policy loss: {policyLoss.item():.4f}")
                else:
                    info(f"Ended with losses: \n - policy loss: {policyLoss.item():.4f}")
                    
    if (mctsGPU == False and trainingOnGPU == True):
        model.cpu()

    return model

def train (model: Net, startingIteration: int):
    """ Trains the model using self play & evaluation """
    datasets = loadDataset() # Load the dataset if it's present in the directory
    
    for iteration in range(startingIteration, iterations + startingIteration):
        print(f"\nAt iteration: {iteration + 1} / {iterations + startingIteration}")
        info(f"At iteration: {iteration + 1}:")
        
        # Create new dataset and append it to datasets
        states, probs, rewards = createDataset(model, iteration - startingIteration)
        datasets.append([states, probs, rewards])
        
        # Remove old datapoints
        while (len(datasets) > window(iteration + 1)):
            datasets.pop(0)
        
        # Concatenate states, probs & rewards for training 
        data = [[] for _ in range(3)]
        for d in datasets:
            for idx, val in enumerate(d):
                data[idx].extend(val)
                
        data = [np.stack(d) for d in data]
        info(f"There is currently {data[0].shape[0]} datapoints in training data, from a maximum of {window(iteration)} games")
        states, probs, rewards = convertTrainingDataToTensors(data)
        
        model = updateWeights(model, states, probs, rewards)

        # Evaluate the model and get the percentage of games won by the new model
        percentage = evaluateModel(model, iteration) 
        
        if (percentage >= 50.0):
            saveModel(model, f"{iteration + 1}.pt")
        else:
            datasets.pop() # Remove the last entry from the dataset 
            model = loadLatetestModel()[0] # Load better model
            
        info("\n")
    
    # Save the dataest for the future if needed
    saveDataset(datasets) 

if __name__ == '__main__':
    model, iteration = loadLatetestModel()
    if (iteration == 0):
        saveModel(model, "0.pt")
        
    train(model, iteration)