import re
from typing import List, Tuple 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os 
from utils import fileNumbersInDirectory
from elo import loadEloes

def getEloes () -> Tuple[np.array]:
    """ Load the eloes and split them into three numpy arrays """
    eloes = loadEloes()
    
    policyAndValueHead, policyHead = [], []
    keys = fileNumbersInDirectory("models")
    
    for key in keys:
        policyAndValueHead.append(eloes[str(key)]["p+v"])
        policyHead.append(eloes[str(key)]["p"])     
    
    return (np.array([int(key) for key in keys]), np.array(policyAndValueHead), np.array(policyHead))

def getNumber (line: str) -> float:
    """ Gets a number from the line """
    return float(re.findall("\d+\.\d+", line)[0])

def getLosses () -> List[np.array]:
    """ Load losses from the info.txt file (under logs) """
    with open("logs/info.txt", "r") as file:
        # Read data from file
        contents = file.read() 
        lines = contents.split("\n")
        linesWithLoss = []
        for l in lines:
            if (len(l) > 0 and l[0] == " "):
                linesWithLoss.append(l)
                
        # Only save the numbers
        losses = []
        for idx in range(len(linesWithLoss) // 2):
            losses.append((getNumber(linesWithLoss[2 * idx]),
                          getNumber(linesWithLoss[2 * idx + 1])))
        
        return [np.array([l[idx] for l in losses]) for idx in range(2)]

def plotData ():
    """ Plots the losses against the epoch """
    plt.style.use("ggplot")
    gs = gridspec.GridSpec(2, 2)
    
    losses = getLosses()
    eloes = getEloes()
    
    # fig.suptitle("Loss / Epoch")
    # Plot value loss
    ax = plt.subplot(gs[0, 0])
    ax.set_title("Value Loss")
    ax.plot(np.arange(len(losses[0])), losses[0], "tab:green")
    ax.set(xlabel = "Iteration", ylabel = "Loss")
    
    # Plot policy loss
    ax = plt.subplot(gs[1, 0])
    ax.set_title("Policy Loss")
    ax.plot(np.arange(len(losses[1])), losses[1], "tab:orange")
    ax.set(xlabel = "Iteration", ylabel = "Loss")

    ax = plt.subplot(gs[:, 1:])
    ax.set_title("Elo versus Iteration")
    eloes = getEloes()
    ax.plot(eloes[0], eloes[1], "tab:blue")
    ax.plot(eloes[0], eloes[2], "tab:red")
    plt.legend(["Policy & Value", "Policy only"])
    ax.set(xlabel = "Iteration", ylabel = "Elo")
    plt.show()
    
def fileNumbersInDirectory (folder: str) -> List[str]:
    """ Returns a list of files in the folder """
    files = os.listdir(folder)
    numbers = [] 
    for f in files:
        numbers.append(int(f.split(".")[0]))
    return sorted(numbers)

if (__name__ == "__main__"):
    plotData()