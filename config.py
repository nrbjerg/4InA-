from math import floor

# MCTS: 
Cpuct = 1.0
mctsGPU = True
epsilon = 1e-8

# Training:
iterations = 200
numberOfGames = 10
rooloutsDuringTraining = 24
tau = 8 # After this many moves, the moves will be deterministic.

# Enable value head at this 
enableValueHeadAfterIteration = iterations - (iterations // 2)

# Training window
def window (iteration: int) -> int:
    if (iteration < 8):
        return 2
    else:
        return int(round(iteration ** (1.0 / 3)))

learningRate = 0.001 
epochs = 10
batchSize = 64
trainingOnGPU = True

# Evaluation:
numberOfEvaluationGames = 49 # Per player (max 49.)
rooloutsDuringEvaluation = 16

# Model:
numberOfFilters = 128
numberOfResidualBlocks = 16
numberOfNeurons = 256 # In the heads of the networks 
performBatchNorm = True
dropoutRate = 0.3
disableValueHead = False

# Valuehead:
valueHeadFilters = 8

# Policyhead:
policyHeadFilters = 32

# State: 
height, width = 6, 7
numberOfMapsPerPlayer = 2