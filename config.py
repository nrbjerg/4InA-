# MCTS: 
Cpuct = 3.0
mctsGPU = True
epsilon = 1e-8

# Training:
iterations = 20
numberOfGames = 20
rooloutsDuringTraining = 200
# Enable value head halfway through the training process
enableValueHeadAfterIteration = iterations // 2

# Training window
def Window (iteration: int) -> int:
    if (iteration < 3):
        return 2
    else:
        return iteration // 2

learningRate = 0.001 # TODO: Implement adaptive learning rate
epochs = 8
batchSize = 100
trainingOnGPU = True

# Evaluation:
numberOfEvaluationGames = 49 # Per player (max 49.)
rooloutsDuringEvaluation = 32

# Model:
numberOfFilters = 128 
numberOfResidualBlocks = 4 # TRAINING: INCREASE
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
# Can be set to something other than 1 if the model should receive old maps
numberOfMapsPerPlayer = 1 