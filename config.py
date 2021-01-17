
# ------------------------------- MTCS configurations --------------------------------------
mctsSimulations = 25
c = 1 # Exploration / Exploitation



# ----------------------------- Training configurations ------------------------------------
epochs = 40 # Number of epochs pr iteration
newGamesPerIteration = 100 # Number of new games per iteration
savedEpisodes = 10000 # Number of saved episodes used for training
batchSize = 500
iterations = 1
learningRate = 0.001 # Recommended learning rate is 0.001
gpu = True



# ------------------------------ Model configurations --------------------------------------
numberOfResidualBlocks = 4 # TODO: This can be increased
numberOfChannels = 64 # TODO: Increase this for actual model # Number of channels in residual blocks
dropoutRate = 0.1



