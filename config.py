import torch 

# MCTS: 
Cpuct = 1.0
mctsGPU = bool(torch.cuda.is_available())
epsilon = 1e-8

# Training:
iterations = 12
numberOfGames = 32
rooloutsDuringTraining = 128

tau = 8 # After this many moves, the moves will be deterministic.
customReward = True
rewardDropOf = 30

# Enable value head at this 
enableValueHeadAfterIteration = 0 #iterations - (iterations // 2)

# Training window
def window (iteration: int) -> int:
    if (iteration < 8):
        return 2
    else:
        return int(round(iteration ** (1.0 / 3)))

learningRate = 0.001 
epochs = 5 
batchSize = 64
trainingOnGPU = bool(torch.cuda.is_available()) 

# Evaluation:
numberOfEvaluationGames = 49 # Per player (max 49.)
rooloutsDuringEvaluation = 16

# Model:
numberOfFilters = 128
numberOfResidualBlocks = 6
numberOfNeurons = 256 # In the heads of the networks 
numberOfHiddenLayers = 3 
performBatchNorm = True
dropoutRate = 0.2
disableValueHead = False # NOTE: This may be overwritten by the variable enableValueHeadAfterIteration

# Valuehead:
valueHeadFilters = 8

# Policyhead:
policyHeadFilters = 32

# State: 
height, width = 6, 7
numberOfMapsPerPlayer = 1 # Right now it can only be 1 TODO: Implement this again ;) 
