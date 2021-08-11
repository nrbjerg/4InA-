"""import torch

# MCTS:
Cpuct = 0.5
mctsGPU = bool(torch.cuda.is_available())
epsilon = 1e-8

# Training:
iterations = 4
numberOfGames = 100
rooloutsDuringTraining = 24 

# Data collection
tau = 12 # After this many moves, the moves will be deterministic.
customReward = False # Use a custom reward function
rewardDropOf = 18 # Assign rewards to this many positions (ie. if the game was won on move 32, assign rewards to state 22 - 32 (if rewardDropOf is set to 10))

# Enable value head at this 
enableValueHeadAfterIteration = 0 # iterations - (iterations // 2)

# Training window
def window (iteration: int) -> int:
    if (iteration < 8):
        return 2
    else:
        return int(round(iteration ** (1.0 / 3)))

learningRate = 0.003
epochs = 20
batchSize = 128
trainingOnGPU = bool(torch.cuda.is_available())

# Evaluation:
numberOfEvaluationGames = 49 # Per player (7 or 49)
rooloutsDuringEvaluation = 16

# Model:
numberOfFilters = 128
numberOfResidualBlocks = 4
numberOfNeurons = 128 # In the heads of the networks
numberOfHiddenLayers = 2
performBatchNorm = True
dropoutRate = 0.2

# Valuehead:
valueHeadFilters = 16

# Policyhead:
policyHeadFilters = 48

# State: 
height, width = 6, 7
numberOfMapsPerPlayer = 1
"""
import torch 

# MCTS: 
Cpuct = 1.0
mctsGPU = bool(torch.cuda.is_available())
epsilon = 1e-8

# Training:
iterations = 12
numberOfGames = 100
rooloutsDuringTraining = 24
tau = 8 # After this many moves, the moves will be deterministic.
customReward = False
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
batchSize = 256
trainingOnGPU = bool(torch.cuda.is_available()) 

# Evaluation:
numberOfEvaluationGames = 49 # Per player (max 49.)
rooloutsDuringEvaluation = 16

# Model:
numberOfFilters = 144
numberOfResidualBlocks = 8
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
numberOfMapsPerPlayer = 2 
