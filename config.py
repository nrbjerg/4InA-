import torch

# MCTS:
Cpuct = 0.5
mctsGPU = True if (torch.cuda.is_available()) else False
epsilon = 1e-8

# Training:
iterations = 100
numberOfGames = 2
rooloutsDuringTraining = 48

# Data collection
tau = 12 # After this many moves, the moves will be deterministic.
customReward = True # Use a custom reward function
rewardDropOf = 10 # After assign rewards this many moves back

# Enable value head at this 
enableValueHeadAfterIteration = 0 # iterations - (iterations // 2)

# Training window
def window (iteration: int) -> int:
    if (iteration < 8):
        return 2
    else:
        return int(round(iteration ** (1.0 / 3)))

learningRate = 0.001 
epochs = 10
batchSize = 128
trainingOnGPU = True if (torch.cuda.is_available()) else False

# Evaluation:
numberOfEvaluationGames = 49 # Per player (max 49.)
rooloutsDuringEvaluation = 16

# Model:
numberOfFilters = 256
numberOfResidualBlocks = 6
numberOfNeurons = 512 # In the heads of the networks 
numberOfHiddenLayers = 2
performBatchNorm = True
dropoutRate = 0.2
disableValueHead = False

# Valuehead:
valueHeadFilters = 8

# Policyhead:
policyHeadFilters = 32

# State: 
height, width = 6, 7
numberOfMapsPerPlayer = 1
